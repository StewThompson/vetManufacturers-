import os
import re
from typing import Callable
from dotenv import load_dotenv
from google import genai

from src.models.manufacturer import Manufacturer
from src.models.assessment import RiskAssessment
from src.data_retrieval.osha_client import OSHAClient
from src.scoring.risk_assessor import RiskAssessor

load_dotenv()

class VettingAgent:
    _NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")

    def __init__(self):
        self.osha_client = OSHAClient()
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

    def vet_by_raw_estab_names(
        self,
        raw_names: list[str],
        display_name: str,
        years_back: int = 0,
        progress_cb: Callable[[str], None] | None = None,
    ) -> "RiskAssessment":
        """
        Run a risk assessment directly against a specific list of raw OSHA
        establishment names (e.g. individual facilities selected by the user).
        This bypasses the normal name-resolution path.
        """
        if progress_cb:
            progress_cb("🔍 Searching OSHA records…")
        self.osha_client.ensure_cache()

        # Collect all inspections for the given raw estab keys, deduplicated by activity_nr.
        seen_activity_nrs: set = set()
        all_inspections = []

        if self.osha_client._use_sqlite:
            for raw in raw_names:
                # Query by estab_name directly so only the selected facility's
                # inspections are returned, not every facility sharing the same
                # company_key across all states.
                rows = self.osha_client._db_rows(
                    "SELECT * FROM inspections WHERE estab_name = ? COLLATE NOCASE",
                    (raw.upper(),),
                )
                for row in rows:
                    act_nr = str(row.get("activity_nr", ""))
                    if act_nr and act_nr not in seen_activity_nrs:
                        seen_activity_nrs.add(act_nr)
                        all_inspections.append(row)
        else:
            for raw in raw_names:
                # _inspections_by_estab is keyed by uppercase raw estab_name.
                for insp in self.osha_client._inspections_by_estab.get(raw.upper(), []):
                    act_nr = str(insp.get("activity_nr", ""))
                    if act_nr and act_nr not in seen_activity_nrs:
                        seen_activity_nrs.add(act_nr)
                        all_inspections.append(insp)

        print(f"  vet_by_raw_estab_names: {len(raw_names)} selected name(s) → "
              f"{len(seen_activity_nrs)} unique inspections")
        if progress_cb:
            progress_cb("🏗 Building inspection records…")
        records = self.osha_client._build_records(all_inspections, years_back=years_back) if all_inspections else []
        if progress_cb:
            progress_cb("🤖 Scoring risk…")
        manufacturer = Manufacturer(name=display_name)
        assessment = self.risk_assessor.assess(manufacturer, records)
        return assessment

    def vet_manufacturer(
        self,
        name: str,
        location: str = None,
        locations: list[str] = None,
        years_back: int = 0,
        progress_cb: Callable[[str], None] | None = None,
    ) -> RiskAssessment:
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
        if progress_cb:
            progress_cb("🔍 Searching OSHA records…")
        if locations:
            self.osha_client.ensure_cache()
            records = self.osha_client._search_cache(name, locations, years_back=years_back)
            if records is None:
                records = []
        else:
            records = self.osha_client.search_manufacturer(manufacturer, years_back=years_back)

        if progress_cb:
            progress_cb("🏗 Building inspection records…")
        # (record building happens inside search_manufacturer / _search_cache)

        # 3. Assess risk
        if progress_cb:
            progress_cb("🤖 Scoring risk…")
        assessment = self.risk_assessor.assess(manufacturer, records)

        # LLM enhancement is intentionally deferred — caller decides when to invoke it
        return assessment

    def enhance_explanation(self, assessment: RiskAssessment):
        """
        Uses Gemini to generate a more natural language summary of the risk
        and translate technical OSHA standards into plain English.
        """
        try:
            prompt = f"""
            You are a manufacturing compliance expert. Review the following risk assessment data.
            The user is a procurement officer with NO legal background. They need to understand the practical implications.

            Manufacturer: {assessment.manufacturer.name}
            Risk Score: {assessment.risk_score}/100
            Recommendation: {assessment.recommendation}
            
            Current Technical Findings (Raw Data):
            {assessment.explanation}
            
            TASK: EXECUTIVE SUMMARY
            Write a 2-3 sentence executive summary explaining the primary drivers of this risk score.

            Format the output clearly with sections:
            ### Executive Summary
            ...
            """
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            if response.text:
                assessment.explanation = response.text
                
        except Exception as e:
            print(f"Error generating LLM summary: {e}")

    @classmethod
    def _normalize_code_text(cls, value: str) -> str:
        """Normalize OSHA code fragments for robust matching across formats."""
        if not value:
            return ""
        return cls._NON_ALNUM_RE.sub("", str(value).upper())

    @classmethod
    def _extract_code_like_tokens(cls, question: str) -> list[str]:
        """Extract likely code/citation tokens from a natural-language question."""
        if not question:
            return []

        raw_tokens = re.split(r"[^A-Z0-9]+", question.upper())
        out: list[str] = []
        for tok in raw_tokens:
            if len(tok) < 3:
                continue
            if not any(ch.isdigit() for ch in tok):
                continue
            # Keep OSHA-like standards (1910/1926...), citation fragments (B01), etc.
            if (
                tok.startswith(("19", "29"))
                or re.fullmatch(r"[A-Z]{1,3}\d{1,4}[A-Z]{0,3}", tok)
                or re.fullmatch(r"\d{2,4}[A-Z]{1,3}", tok)
            ):
                out.append(tok)
        # Preserve order while de-duplicating
        seen: set[str] = set()
        deduped: list[str] = []
        for tok in out:
            if tok not in seen:
                seen.add(tok)
                deduped.append(tok)
        return deduped

    def get_code_evidence_report(self, assessment: RiskAssessment, question: str) -> str:
        """
        Build a deterministic evidence block for OSHA standard/citation questions.

        Matches standards and citation IDs in normalized form so a query like
        "1910.0030 B01" can match cached data stored as "19100030 B01".
        """
        if not assessment.records:
            return "No OSHA inspection records available for code-specific lookup."

        q_norm = self._normalize_code_text(question)
        q_tokens = self._extract_code_like_tokens(question)

        def _matches(norm_value: str, raw_value: str) -> bool:
            if not norm_value:
                return False
            if norm_value in q_norm or q_norm in norm_value:
                return True
            for tok in q_tokens:
                norm_tok = self._normalize_code_text(tok)
                if not norm_tok:
                    continue
                if norm_tok in norm_value or norm_value in norm_tok:
                    return True
                # Also compare against raw value with whitespace stripped
                if norm_tok in self._normalize_code_text(raw_value):
                    return True
            return False

        matches: list[dict] = []
        seen_standards: set[str] = set()

        for r in assessment.records:
            gen_duty_notes_for_inspection: list[str] = []
            for gv in r.violations:
                if not self.osha_client._is_gen_duty_standard(gv.category):
                    continue
                citation_id = gv.citation_id or ""
                if not citation_id:
                    continue
                # Pull inspector narrative directly so we include notes beyond
                # the high-priority subset attached during record build.
                gd_note = self.osha_client.get_gen_duty_narrative(r.inspection_id, citation_id)
                if gd_note:
                    gen_duty_notes_for_inspection.append(
                        f"Citation {citation_id}: {gd_note.strip()}"
                    )

            for v in r.violations:
                norm_std = self._normalize_code_text(v.category)
                norm_cit = self._normalize_code_text(v.citation_id or "")
                if _matches(norm_std, v.category) or _matches(norm_cit, v.citation_id or ""):
                    seen_standards.add(v.category)
                    match_row = {
                        "inspection_id": r.inspection_id,
                        "date_opened": str(r.date_opened),
                        "estab_name": r.estab_name or "UNKNOWN",
                        "site_city": r.site_city or "",
                        "site_state": r.site_state or "",
                        "standard": v.category,
                        "citation_id": v.citation_id or "N/A",
                        "severity": v.severity,
                        "penalty": v.penalty_amount,
                        "gravity": v.gravity or "",
                        "nr_exposed": v.nr_exposed,
                        "hazardous_substance": v.hazardous_substance or "",
                        "gen_duty_narrative": (v.gen_duty_narrative or "").strip(),
                        "gen_duty_notes_all": gen_duty_notes_for_inspection,
                        "accidents": r.accidents,
                    }
                    matches.append(match_row)

        if not q_tokens and not any(ch.isdigit() for ch in q_norm):
            return (
                "No code-like token detected in the question. "
                "Ask with a standard/citation such as '19100030 B01' or '01001A'."
            )

        if not matches:
            # Provide nearby standards as a hint when possible.
            all_standards = sorted({v.category for rec in assessment.records for v in rec.violations})
            hints: list[str] = []
            for tok in q_tokens:
                tok_norm = self._normalize_code_text(tok)
                if len(tok_norm) < 4:
                    continue
                prefix = tok_norm[:6]
                for std in all_standards:
                    if self._normalize_code_text(std).startswith(prefix):
                        hints.append(std)
                    if len(hints) >= 8:
                        break
                if hints:
                    break

            hint_block = ""
            if hints:
                hint_block = "\nNearby standards in this company's data: " + ", ".join(hints)

            tok_text = ", ".join(q_tokens) if q_tokens else "(none parsed)"
            return (
                "No exact matching citation/standard found in this company's OSHA records."
                f"\nParsed query tokens: {tok_text}"
                f"{hint_block}"
            )

        total_penalty = sum(m["penalty"] for m in matches)
        out = [
            "Code Evidence Retrieval Result",
            f"Query tokens: {', '.join(q_tokens) if q_tokens else '(none parsed)'}",
            f"Matched violation rows: {len(matches)}",
            f"Matched standards: {', '.join(sorted(seen_standards))}",
            f"Total penalty across matches: ${total_penalty:,.2f}",
            "",
            "Evidence rows:",
        ]

        # Keep prompt context bounded to avoid token blow-up.
        for i, m in enumerate(matches[:20], start=1):
            loc = ", ".join(p for p in [m["site_city"], m["site_state"]] if p) or "Unknown location"
            out.append(
                f"{i}. Insp {m['inspection_id']} ({m['date_opened']}, {loc}) | "
                f"Std {m['standard']} | Citation {m['citation_id']} | "
                f"{m['severity']} | Penalty ${m['penalty']:,.2f}"
            )
            if m["gravity"]:
                out.append(f"   Gravity: {m['gravity']}")
            if m["nr_exposed"] is not None:
                out.append(f"   Exposed workers: {m['nr_exposed']}")
            if m["hazardous_substance"]:
                out.append(f"   Hazardous substance(s): {m['hazardous_substance']}")
            if m["gen_duty_narrative"]:
                snippet = m["gen_duty_narrative"][:500]
                if len(m["gen_duty_narrative"]) > 500:
                    snippet += "…"
                out.append(f"   Inspector notes: {snippet}")
            if m["gen_duty_notes_all"]:
                out.append("   General Duty notes on this inspection:")
                for note in m["gen_duty_notes_all"][:3]:
                    note_snippet = note[:500]
                    if len(note) > 500:
                        note_snippet += "…"
                    out.append(f"   - {note_snippet}")
                if len(m["gen_duty_notes_all"]) > 3:
                    out.append(
                        f"   - ... {len(m['gen_duty_notes_all']) - 3} additional Gen Duty note(s) omitted."
                    )
            if m["accidents"]:
                out.append(f"   Linked accidents on inspection: {len(m['accidents'])}")

        if len(matches) > 20:
            out.append(f"... {len(matches) - 20} additional matched rows omitted for brevity.")

        return "\n".join(out)

    def discuss_assessment(self, assessment: RiskAssessment, question: str) -> str:
        """
        Interactive Q&A layer using Gemini.
        Includes accident/injury context so users can drill into specific incidents.
        """
        if not self.client:
            return "LLM features are not available. Please set GOOGLE_API_KEY."

        print(f"User asked: '{question}' about {assessment.manufacturer.name}")

        try:
            code_evidence_context = self.get_code_evidence_report(assessment, question)

            # Build accident detail context
            accident_context = ""
            for r in assessment.records:
                if r.accidents:
                    for acc in r.accidents:
                        fat_tag = " [FATALITY]" if acc.fatality else ""
                        accident_context += f"\nAccident {acc.summary_nr}{fat_tag} ({acc.event_date or 'unknown'}):\n"
                        accident_context += f"  Description: {acc.event_desc}\n"
                        for inj in acc.injuries:
                            nature = inj.get('nature', 'Not reported')
                            body = inj.get('body_part', 'Not reported')
                            degree = inj.get('degree', 'Not reported')
                            event_type = inj.get('event_type', '')
                            et_suffix = f" [{event_type}]" if event_type and event_type not in ('Not reported', '') else ""
                            accident_context += f"  Injury: {nature} to {body} — {degree}{et_suffix}"
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

            Code/Citation Evidence Retrieval (normalized standard + citation matching):
            {code_evidence_context}
            
            User Question: {question}
            
            Answer the user's question based on the provided assessment data and accident details.
            Detailed Instructions:
            - If the user asks about a specific citation or code, prioritize the 'Code/Citation Evidence Retrieval' section and cite matched inspection IDs, standard, citation ID, and penalties.
            - If the user asks about a specific standard (e.g., '1910.1200'), explain what it is in plain English and why it's a safety risk.
            - If the user asks about accidents or injuries, use the 'Accident & Injury Details' section. Describe the incident, injuries, and any fatalities clearly.
            - If the user asks about General Duty violations or specific hazards/safety problems, use the 'General Duty Clause Inspector Notes' section to give specific, plain-language answers about what the inspector actually found.
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
