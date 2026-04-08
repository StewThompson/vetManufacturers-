import os
import json
import logging
import re
from typing import Callable
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from src.models.manufacturer import Manufacturer
from src.models.assessment import RiskAssessment
from src.data_retrieval.osha_client import OSHAClient
from src.scoring.risk_assessor import RiskAssessor

load_dotenv()

logger = logging.getLogger(__name__)

class VettingAgent:
    _NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")

    def __init__(self):
        self.osha_client = OSHAClient()
        self.risk_assessor = RiskAssessor(osha_client=self.osha_client)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            logger.warning("GOOGLE_API_KEY not found. LLM features will be disabled.")
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

    def reassess(self, manufacturer: Manufacturer, records: list) -> "RiskAssessment":
        """Re-run scoring on a pre-filtered record set without fetching new data."""
        return self.risk_assessor.assess(manufacturer, records)

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

        logger.info(
            "  vet_by_raw_estab_names: %s selected name(s) → %s unique inspections",
            len(raw_names), len(seen_activity_nrs),
        )
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
        logger.info("Agent starting vetting process for: %s", name)

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
            rt = assessment.risk_targets
            targets_block = ""
            if rt:
                targets_block = f"""
            Multi-Target ML Predictions (12-month forward outlook):
              Composite Risk Score: {rt.composite_risk_score}/100
              P(Serious/Willful/Repeat event): {rt.p_serious_wr_event:.1%}
              Expected Penalty: ${rt.expected_penalty_usd_12m:,.0f}
              P(Penalty ≥ NAICS P75): {rt.p_penalty_ge_p75:.1%}
              P(Penalty ≥ NAICS P90 / ${rt.industry_p90_penalty:,.0f}): {rt.p_penalty_ge_p90:.1%}
              P(Penalty ≥ NAICS P95): {rt.p_penalty_ge_p95:.1%}
"""

            prompt = f"""
            You are a manufacturing compliance expert. Review the following risk assessment data.
            The user is a procurement officer with NO legal background. They need to understand the practical implications.

            Manufacturer: {assessment.manufacturer.name}
            Risk Score: {assessment.risk_score}/100
            Recommendation: {assessment.recommendation}
            {targets_block}
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
            logger.warning("Error generating LLM summary: %s", e)

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

    # ------------------------------------------------------------------ #
    #  Gemini tool definitions
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_osha_tools() -> "genai_types.Tool":
        """Return a Gemini Tool with OSHA data-retrieval function declarations."""
        return genai_types.Tool(function_declarations=[
            genai_types.FunctionDeclaration(
                name="fetch_gen_duty_narratives",
                description=(
                    "Retrieve the full OSHA inspector narrative text for one or more "
                    "General Duty clause citations. Use this when the user asks about "
                    "what an inspector actually found, specific hazards described, or the "
                    "full text of a citation. Correct any typos in inspection IDs by "
                    "matching against the known inspection list."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": (
                                "List of objects, each with 'inspection_id' and 'citation_id'. "
                                "Example: [{\"inspection_id\": \"315477917\", \"citation_id\": \"01001A\"}]"
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "inspection_id": {"type": "string"},
                                    "citation_id": {"type": "string"},
                                },
                                "required": ["inspection_id", "citation_id"],
                            },
                        }
                    },
                    "required": ["items"],
                },
            ),
            genai_types.FunctionDeclaration(
                name="fetch_accident_abstracts",
                description=(
                    "Retrieve the full OSHA accident abstract (detailed fatality/injury "
                    "investigation narrative) for one or more accident summary numbers. "
                    "Use this when the user asks for full details of a workplace accident, "
                    "fatality description, or investigation findings. Correct typos by "
                    "matching against the known accident summary list."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary_nrs": {
                            "type": "array",
                            "description": "List of accident summary_nr strings.",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["summary_nrs"],
                },
            ),
            genai_types.FunctionDeclaration(
                name="fetch_inspection_violations",
                description=(
                    "Retrieve all violation records for one or more inspection IDs "
                    "(activity_nr). Use this to get the complete list of violations, "
                    "standards cited, and penalty amounts for a specific inspection. "
                    "Correct typos by matching against the known inspection ID list."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "activity_nrs": {
                            "type": "array",
                            "description": "List of inspection activity_nr strings.",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["activity_nrs"],
                },
            ),
        ])

    def _dispatch_tool_call(
        self, fn_name: str, fn_args: dict, assessment: "RiskAssessment"
    ) -> str:
        """Execute a Gemini function call and return the result as a string."""
        # Collect known IDs from assessment for typo-correction (fuzzy match)
        known_insp_ids = [r.inspection_id for r in assessment.records if r.inspection_id]
        known_summary_nrs = [
            acc.summary_nr
            for r in assessment.records
            for acc in r.accidents
            if acc.summary_nr
        ]

        def _best_match(query: str, candidates: list[str]) -> str:
            """Return closest candidate by prefix/substring, or original."""
            q = query.strip()
            # Exact first
            if q in candidates:
                return q
            # Prefix match
            for c in candidates:
                if c.startswith(q) or q.startswith(c):
                    return c
            # Substring
            for c in candidates:
                if q in c or c in q:
                    return c
            return q  # fallback: use as-is

        if fn_name == "fetch_gen_duty_narratives":
            items = fn_args.get("items", [])
            results = []
            for item in items:
                insp_id = _best_match(str(item.get("inspection_id", "")), known_insp_ids)
                cit_id = str(item.get("citation_id", ""))
                narrative = self.osha_client.get_gen_duty_narrative(insp_id, cit_id)
                results.append(
                    f"Inspection {insp_id} / Citation {cit_id}:\n"
                    + (narrative if narrative else "(No narrative found)")
                )
            return "\n\n".join(results) if results else "No items requested."

        elif fn_name == "fetch_accident_abstracts":
            summary_nrs = fn_args.get("summary_nrs", [])
            results = []
            for snr in summary_nrs:
                matched = _best_match(str(snr), known_summary_nrs)
                abstract = self.osha_client.get_accident_abstract(matched)
                results.append(
                    f"Accident {matched}:\n"
                    + (abstract if abstract else "(No abstract found)")
                )
            return "\n\n".join(results) if results else "No summary numbers requested."

        elif fn_name == "fetch_inspection_violations":
            activity_nrs = fn_args.get("activity_nrs", [])
            results = []
            for act_nr in activity_nrs:
                matched = _best_match(str(act_nr), known_insp_ids)
                viols = self.osha_client.get_violations_for_activity(matched)
                if not viols:
                    results.append(f"Inspection {matched}: No violations found.")
                    continue
                lines = [f"Inspection {matched} — {len(viols)} violation(s):"]
                for v in viols[:30]:
                    std = v.get("standard", v.get("citation_id", "?"))
                    vtype = v.get("viol_type", "")
                    pen = v.get("current_penalty", v.get("initial_penalty", "0"))
                    lines.append(f"  • Std {std} | Type {vtype} | Penalty ${pen}")
                if len(viols) > 30:
                    lines.append(f"  ... {len(viols) - 30} more violations omitted.")
                results.append("\n".join(lines))
            return "\n\n".join(results) if results else "No activity numbers requested."

        return f"Unknown function: {fn_name}"

    def discuss_assessment(self, assessment: RiskAssessment, question: str) -> str:
        """
        Interactive Q&A layer using Gemini with function-calling tools for
        on-demand OSHA data retrieval (gen-duty narratives, accident abstracts,
        inspection violations).
        """
        if not self.client:
            return "LLM features are not available. Please set GOOGLE_API_KEY."

        logger.debug("User asked: '%s' about %s", question, assessment.manufacturer.name)

        try:
            code_evidence_context = self.get_code_evidence_report(assessment, question)

            # ── Build static context ──────────────────────────────────
            accident_context = ""
            for r in assessment.records:
                for acc in r.accidents:
                    fat_tag = " [FATALITY]" if acc.fatality else ""
                    accident_context += (
                        f"\nAccident {acc.summary_nr}{fat_tag} ({acc.event_date or 'unknown'}):\n"
                        f"  Description: {acc.event_desc}\n"
                    )
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
                    if acc.summary_nr and not acc.abstract:
                        abstract = self.osha_client.get_accident_abstract(acc.summary_nr)
                        if abstract:
                            accident_context += f"  Full Abstract: {abstract[:1000]}\n"
            if not accident_context:
                accident_context = "No accident/injury records linked to this manufacturer."

            gen_duty_context = ""
            for r in assessment.records:
                for v in r.violations:
                    if getattr(v, "gen_duty_narrative", None):
                        pen = f"${v.penalty_amount:,.0f}" if v.penalty_amount else "N/A"
                        gen_duty_context += (
                            f"\nInspection {r.inspection_id} / Citation {v.citation_id or 'N/A'}"
                            f" (Penalty {pen}, {v.severity}):\n  {v.gen_duty_narrative}\n"
                        )
            if not gen_duty_context:
                gen_duty_context = "None attached; use fetch_gen_duty_narratives tool if needed."

            # ── Multi-target predictions block ────────────────────────
            rt = assessment.risk_targets
            targets_block = ""
            if rt:
                targets_block = f"""
Multi-Target ML Predictions (12-month forward outlook):
  Composite Risk Score : {rt.composite_risk_score}/100
  P(S/WR event)        : {rt.p_serious_wr_event:.1%}
  Expected Penalty     : ${rt.expected_penalty_usd_12m:,.0f}
  P(≥ NAICS P75)       : {rt.p_penalty_ge_p75:.1%}
  P(≥ NAICS P90 / ${rt.industry_p90_penalty:,.0f}): {rt.p_penalty_ge_p90:.1%}
  P(≥ NAICS P95)       : {rt.p_penalty_ge_p95:.1%}
"""

            # ── Known IDs index for Gemini reference ──────────────────
            known_inspection_ids = list({r.inspection_id for r in assessment.records if r.inspection_id})
            known_summary_nrs = list({
                acc.summary_nr
                for r in assessment.records
                for acc in r.accidents
                if acc.summary_nr
            })

            system_prompt = f"""You are a helpful AI assistant for a manufacturing procurement vetting platform.

Manufacturer: {assessment.manufacturer.name}
Risk Score: {assessment.risk_score}/100
Recommendation: {assessment.recommendation}
{targets_block}
Violation History & Summary:
{assessment.explanation}

Accident & Injury Details (summary):
{accident_context}

General Duty Clause Inspector Notes (pre-loaded subset):
{gen_duty_context}

Code/Citation Evidence Retrieval:
{code_evidence_context}

KNOWN DATA INDEXES (use these to resolve typos or approximate references from the user):
  Inspection IDs (activity_nr): {json.dumps(known_inspection_ids[:100])}
  Accident summary numbers: {json.dumps(known_summary_nrs)}

TOOL USE INSTRUCTIONS:
You have three retrieval tools available:
  1. fetch_gen_duty_narratives — call with a list of {{inspection_id, citation_id}} objects to retrieve
     the full plain-language inspector narrative for any General Duty citation.
  2. fetch_accident_abstracts — call with a list of summary_nr strings to retrieve the complete OSHA
     accident investigation abstract (fatality descriptions, sequence of events, root cause).
  3. fetch_inspection_violations — call with a list of activity_nr strings to retrieve all violations
     for that inspection with standard codes, types, and penalties.

If the user mentions an inspection ID, citation, or accident number with a possible typo, match it
against the KNOWN DATA INDEXES above and use the closest match when calling tools.

Answer the user's question thoroughly. Use tools proactively when full narrative/abstract detail
would improve the answer. Be professional and concise.
"""

            contents = [system_prompt, f"User Question: {question}"]

            tool = self._make_osha_tools()
            config = genai_types.GenerateContentConfig(tools=[tool])

            # ── Agentic loop: run until no more function calls ─────────
            MAX_ROUNDS = 4
            for _round in range(MAX_ROUNDS):
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config,
                )

                # Collect any function calls in this response
                fn_calls = []
                text_parts = []
                candidate = response.candidates[0] if response.candidates else None
                if candidate:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            fn_calls.append(part.function_call)
                        elif hasattr(part, "text") and part.text:
                            text_parts.append(part.text)

                if not fn_calls:
                    # No tool calls — return the final text
                    return response.text or "".join(text_parts) or "(No response)"

                # Execute each function call and build the function response
                contents.append(candidate.content)
                fn_response_parts = []
                for fc in fn_calls:
                    logger.debug("  [Gemini tool] %s(%s)", fc.name, dict(fc.args))
                    result_text = self._dispatch_tool_call(fc.name, dict(fc.args), assessment)
                    fn_response_parts.append(
                        genai_types.Part.from_function_response(
                            name=fc.name,
                            response={"result": result_text},
                        )
                    )
                contents.append(genai_types.Content(parts=fn_response_parts, role="tool"))

            # Safety fallback: return whatever we have after MAX_ROUNDS
            return response.text or "(No response after tool loop)"

        except Exception as e:
            return f"Error communicating with AI agent: {e}"
