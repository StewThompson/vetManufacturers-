import os
from dotenv import load_dotenv
from google import genai

from src.models.manufacturer import Manufacturer
from src.models.assessment import RiskAssessment
from src.data_retrieval.osha_client import OSHAClient
from src.data_retrieval.reputation_client import ReputationClient
from src.scoring.risk_assessor import RiskAssessor

load_dotenv()

class VettingAgent:
    def __init__(self):
        self.osha_client = OSHAClient()
        self.reputation_client = ReputationClient()
        self.risk_assessor = RiskAssessor(osha_client=self.osha_client)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            print("Warning: GOOGLE_API_KEY not found in environment variables. LLM features will be disabled.")
            self.client = None

    def get_all_company_names(self) -> list[str]:
        """Return branch-deduplicated company names from OSHA cache."""
        return self.osha_client.get_all_company_names()

    def get_locations_for_company(self, company: str) -> list[str]:
        """Return full-address locations recorded for a company."""
        return self.osha_client.get_locations_for_company(company)

    def get_osha_client(self):
        """Expose the underlying OSHAClient so UI helpers can access raw indexes."""
        self.osha_client.ensure_cache()
        return self.osha_client

    def vet_by_raw_estab_names(self, raw_names: list[str], display_name: str) -> "RiskAssessment":
        """
        Run a risk assessment directly against a specific list of raw OSHA
        establishment names (e.g. individual facilities selected by the user).
        This bypasses the normal name-resolution path.
        """
        from src.models.manufacturer import Manufacturer
        self.osha_client.ensure_cache()

        # Collect all inspections for the given raw estab keys
        all_inspections = []
        for raw in raw_names:
            upper = raw.upper()
            all_inspections.extend(self.osha_client._inspections_by_estab.get(upper, []))

        records = self.osha_client._build_records(all_inspections) if all_inspections else []
        manufacturer = Manufacturer(name=display_name)
        reputation_data = self.reputation_client.search_news(display_name)
        assessment = self.risk_assessor.assess(manufacturer, records, reputation_data)
        if self.client:
            self._enhance_explanation(assessment, reputation_data)
        return assessment

    def vet_manufacturer(self, name: str, location: str = None, locations: list[str] = None) -> RiskAssessment:
        """
        Main workflow for vetting a manufacturer.
        1. Resolve identity
        2. Retrieve data
        3. Assess risk
        4. Return assessment
        """
        print(f"Agent starting vetting process for: {name}")
        
        # 1. Resolve identity (stub)
        manufacturer = Manufacturer(name=name, location=location)
        
        # 2. Retrieve data
        if locations:
            self.osha_client.ensure_cache()
            records = self.osha_client._search_cache(name, locations)
            if records is None:
                records = []
        else:
            records = self.osha_client.search_manufacturer(manufacturer)
        
        # 2b. Retrieve reputation data
        reputation_data = self.reputation_client.search_news(manufacturer.name)
        
        # 3. Assess risk
        assessment = self.risk_assessor.assess(manufacturer, records, reputation_data)
        
        # 4. Enhance explanation with LLM (Optional step to summarize findings)
        if self.client:
           self._enhance_explanation(assessment, reputation_data)

        return assessment

    def _enhance_explanation(self, assessment: RiskAssessment, reputation_data: list = None):
        """
        Uses Gemini to generate a more natural language summary of the risk
        and translate technical OSHA standards into plain English, integrating reputation data.
        """
        try:
            feed_data = ""
            if reputation_data:
                feed_data = "Reputation Search Results:\n"
                for i, item in enumerate(reputation_data[:5]):
                    feed_data += f"- {item.get('title')} ({item.get('source')}): {item.get('body')[:200]}...\n"
            else:
                feed_data = "Reputation Search Results: None found.\n"

            prompt = f"""
            You are a manufacturing compliance expert. Review the following risk assessment data.
            The user is a procurement officer with NO legal background. They need to understand the practical implications.

            Manufacturer: {assessment.manufacturer.name}
            Risk Score: {assessment.risk_score}/100 (Reputation Score: {assessment.reputation_score})
            Recommendation: {assessment.recommendation}
            
            Reputation/News Data:
            {feed_data}
            
            Current Technical Findings (Raw Data):
            {assessment.explanation}
            
            TASK 1: EXECUTIVE SUMMARY
            Write a 2-3 sentence executive summary explaining the primary drivers of this risk score. 
            Synthesize the OSHA findings with the reputation data. If there is negative news, mention it.

            TASK 2: REPUTATION ANALYSIS
            Briefly analyze the sentiment of the provided news snippets. Are there any red flags like lawsuits, major accidents, or public scandals not covered by OSHA?
            If no news, say "No significant negative news found."

            Format the output clearly with sections:
            ### Executive Summary
            ...
            ### Reputation Analysis
            ...
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            if response.text:
                assessment.explanation = response.text
                assessment.reputation_summary = "AI Enhanced Summary Included."
                
        except Exception as e:
            print(f"Error generating LLM summary: {e}")

    def discuss_assessment(self, assessment: RiskAssessment, question: str) -> str:
        """
        Interactive Q&A layer using Gemini.
        Includes accident/injury context so users can drill into specific incidents.
        """
        if not self.client:
            return "LLM features are not available. Please set GOOGLE_API_KEY."

        print(f"User asked: '{question}' about {assessment.manufacturer.name}")

        try:
            reputation_context = ""
            if assessment.reputation_data:
                reputation_context = "\nRecent News & Search Results:\n"
                for i, item in enumerate(assessment.reputation_data[:8]):
                   reputation_context += f"{i+1}. {item.get('title')} ({item.get('date')})\n   Source: {item.get('source')}\n   Snippet: {item.get('body')}\n   URL: {item.get('url')}\n\n"
            else:
                reputation_context = "\nNo recent news or reputation data found.\n"

            # Build accident detail context
            accident_context = ""
            for r in assessment.records:
                if r.accidents:
                    for acc in r.accidents:
                        fat_tag = " [FATALITY]" if acc.fatality else ""
                        accident_context += f"\nAccident {acc.summary_nr}{fat_tag} ({acc.event_date or 'unknown'}):\n"
                        accident_context += f"  Description: {acc.event_desc}\n"
                        for inj in acc.injuries:
                            accident_context += f"  Injury: {inj.get('nature', 'Unknown')} to {inj.get('body_part', 'Unknown')} — {inj.get('degree', 'Unknown')}"
                            if inj.get("age"):
                                accident_context += f", age {inj['age']}"
                            accident_context += "\n"
                        # Load abstract on demand if user asks about an accident
                        if acc.summary_nr and not acc.abstract:
                            abstract = self.osha_client.get_accident_abstract(acc.summary_nr)
                            if abstract:
                                accident_context += f"  Full Abstract: {abstract[:1000]}\n"

            if not accident_context:
                accident_context = "\nNo accident/injury records linked to this manufacturer's inspections.\n"

            # Build gen_duty narrative context for on-demand Q&A
            # (auto-summary only shows high-priority ones; LLM gets all attached narratives)
            gen_duty_context = ""
            for r in assessment.records:
                for v in r.violations:
                    if getattr(v, "gen_duty_narrative", None):
                        pen = f"${v.penalty_amount:,.0f}" if v.penalty_amount else "N/A"
                        gen_duty_context += (
                            f"\nInspection {r.inspection_id} / Citation {v.citation_id or 'N/A'}"
                            f" (Penalty {pen}, {v.severity}):\n"
                            f"  {v.gen_duty_narrative}\n"
                        )
            if not gen_duty_context:
                gen_duty_context = "None available for this manufacturer."

            prompt = f"""
            You are a helpful AI assistant for a manufacturing vetting platform. 
            User is asking about the following manufacturer assessment:
            
            Manufacturer: {assessment.manufacturer.name}
            Risk Score: {assessment.risk_score}/100
            Recommendation: {assessment.recommendation}
            
            Violation History & Details:
            {assessment.explanation}

            Accident & Injury Details:
            {accident_context}

            General Duty Clause Inspector Notes (plain-language hazard descriptions written by OSHA inspectors):
            {gen_duty_context}

            {reputation_context}
            
            User Question: {question}
            
            Answer the user's question based on the provided assessment data, accident details, and news snippets. 
            Detailed Instructions:
            - If the user asks about a specific standard (e.g., '1910.1200'), explain what it is in plain English and why it's a safety risk.
            - If the user asks about accidents or injuries, use the 'Accident & Injury Details' section. Describe the incident, injuries, and any fatalities clearly.
            - If the user asks about General Duty violations or specific hazards/safety problems, use the 'General Duty Clause Inspector Notes' section to give specific, plain-language answers about what the inspector actually found.
            - If the user asks about news/reputation, use the 'Recent News & Search Results' section. Cite the source title if relevant.
            - If the answer isn't in the data, say so. 
            - Be professional and concise.
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
            
        except Exception as e:
            return f"Error communicating with AI agent: {e}"
