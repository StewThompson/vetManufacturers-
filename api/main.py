"""
FastAPI backend — wraps VettingAgent, OSHAClient, and grouped_search
behind a clean REST + SSE API.

Start with:
    uvicorn api.main:app --reload --port 8000

Endpoints
---------
GET  /api/health
GET  /api/companies          → sorted list of all company names
GET  /api/search?q=...       → grouped company results
GET  /api/locations?company= → address list for a company
POST /api/assess             → SSE stream: progress events then result
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.vetting_agent import VettingAgent
from src.search.grouped_search import (
    get_or_build_company_key_index,
    group_establishments,
    GroupedCompanyResult,
    FacilityCandidate,
)
from api.schemas import (
    AssessmentResponse,
    ComplianceOutlook12M,
    FacilityOut,
    GroupedCompanyOut,
    OSHARecordOut,
    ProbabilisticRiskTargetsOut,
    SearchResponse,
    SiteScoreOut,
    SSEError,
    SSEProgress,
    SSEResult,
)

# ── App + CORS ────────────────────────────────────────────────────────────────

app = FastAPI(title="Manufacturer Compliance Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (loaded once at startup) ──────────────────────────────────────

_agent: Optional[VettingAgent] = None
_name_index: Optional[dict] = None


@app.on_event("startup")
async def _startup():
    global _agent, _name_index
    _agent = VettingAgent()
    _name_index = get_or_build_company_key_index(_agent.get_osha_client())


def _get_agent() -> VettingAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet")
    return _agent


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _facility_out(f: FacilityCandidate) -> FacilityOut:
    return FacilityOut(
        raw_name=f.raw_name,
        display_name=f.display_name,
        facility_code=f.facility_code,
        city=f.city or "",
        state=f.state or "",
        address=f.address or "",
        naics_code=f.naics_code or "",
        confidence=f.confidence,
        confidence_label=f.confidence_label,
    )


def _group_out(g: GroupedCompanyResult) -> GroupedCompanyOut:
    return GroupedCompanyOut(
        parent_name=g.parent_name,
        total_facilities=g.total_facilities,
        confidence=g.confidence,
        confidence_label=g.confidence_label,
        high_confidence=[_facility_out(f) for f in g.high_confidence],
        medium_confidence=[_facility_out(f) for f in g.medium_confidence],
        low_confidence=[_facility_out(f) for f in g.low_confidence],
    )


def _assessment_response(assessment) -> AssessmentResponse:
    records_out = [
        OSHARecordOut(
            inspection_id=r.inspection_id,
            date_opened=r.date_opened.isoformat(),
            violations=[v.model_dump() for v in r.violations],
            total_penalties=r.total_penalties,
            severe_injury_or_fatality=r.severe_injury_or_fatality,
            accidents=[a.model_dump() for a in r.accidents],
            naics_code=r.naics_code,
            nr_in_estab=r.nr_in_estab,
            estab_name=r.estab_name,
            site_city=r.site_city,
            site_state=r.site_state,
        )
        for r in assessment.records
    ]

    site_scores_out = [
        SiteScoreOut(
            name=s.get("name", ""),
            score=s.get("score", 0.0),
            n_inspections=s.get("n_inspections", 0),
            naics_code=s.get("naics_code"),
            city=s.get("city"),
            state=s.get("state"),
        )
        for s in assessment.site_scores
        # site_scores may contain a private _log_feats array used internally;
        # SiteScoreOut only picks named fields so it never reaches the wire.
    ]

    return AssessmentResponse(
        manufacturer_name=assessment.manufacturer.name,
        risk_score=assessment.risk_score,
        recommendation=assessment.recommendation,
        explanation=assessment.explanation,
        confidence_score=assessment.confidence_score,
        feature_weights=assessment.feature_weights,
        percentile_rank=assessment.percentile_rank,
        industry_label=assessment.industry_label,
        industry_group=assessment.industry_group,
        industry_percentile=assessment.industry_percentile,
        industry_comparison=assessment.industry_comparison,
        missing_naics=assessment.missing_naics,
        establishment_count=assessment.establishment_count,
        site_scores=site_scores_out,
        risk_concentration=assessment.risk_concentration,
        systemic_risk_flag=assessment.systemic_risk_flag,
        aggregation_warning=assessment.aggregation_warning,
        concentration_warning=assessment.concentration_warning,
        records=records_out,
        record_count=len(records_out),
        outlook=(
            ComplianceOutlook12M(**assessment.outlook)
            if assessment.outlook is not None
            else None
        ),
        risk_targets=(
            ProbabilisticRiskTargetsOut(
                p_serious_wr_event=assessment.risk_targets.p_serious_wr_event,
                expected_penalty_usd_12m=assessment.risk_targets.expected_penalty_usd_12m,
                p_injury_event=assessment.risk_targets.p_injury_event,
                gravity_score=assessment.risk_targets.gravity_score,
                composite_risk_score=assessment.risk_targets.composite_risk_score,
            )
            if assessment.risk_targets is not None
            else None
        ),
    )


def _sse(event_type: str, data: dict) -> str:
    """Format a server-sent event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/companies", response_model=List[str])
async def companies():
    """Return sorted, deduplicated list of all company names in the OSHA cache."""
    agent = _get_agent()
    return agent.get_all_company_names()


@app.get("/api/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=2)):
    """Group OSHA establishment names matching a search query."""
    results = group_establishments(query=q, company_key_index=_name_index)
    return SearchResponse(
        query=q,
        top_group=_group_out(results.top_group) if results.top_group else None,
        other_groups=[_group_out(g) for g in results.other_groups],
        unmatched=results.unmatched or [],
    )


@app.get("/api/locations", response_model=List[str])
async def locations(company: str = Query(...)):
    """Return address list for a given company name."""
    agent = _get_agent()
    return agent.get_locations_for_company(company)


@app.get("/api/assess")
async def assess(
    company: Optional[str] = Query(None),
    raw_names: Optional[str] = Query(None, description="Comma-separated raw OSHA names"),
    display_name: Optional[str] = Query(None),
    years_back: int = Query(10),
):
    """
    SSE endpoint — streams progress events then a final result event.

    Use raw_names (comma-separated) to assess specific facilities.
    Use company for a name-resolution based assessment.

    Event types: 'progress', 'result', 'error'
    """
    if not company and not raw_names:
        raise HTTPException(status_code=422, detail="Provide 'company' or 'raw_names'")

    agent = _get_agent()

    async def _stream() -> AsyncGenerator[str, None]:
        progress_msgs: list[str] = []

        def _cb(msg: str):
            progress_msgs.append(msg)

        try:
            import asyncio
            loop = asyncio.get_event_loop()

            # Run the blocking assessment in a thread executor so the event loop
            # stays free to send SSE chunks between progress appends.
            if raw_names:
                names = [n.strip() for n in raw_names.split(",") if n.strip()]
                label = display_name or (names[0] if len(names) == 1 else f"{len(names)} facilities")

                def _work():
                    return agent.vet_by_raw_estab_names(
                        raw_names=names,
                        display_name=label,
                        years_back=years_back,
                        progress_cb=_cb,
                    )
            else:
                label = company

                def _work():
                    return agent.vet_manufacturer(
                        name=company,
                        years_back=years_back,
                        progress_cb=_cb,
                    )

            # Kick off work in a thread
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = loop.run_in_executor(executor, _work)

            # Poll: flush progress messages while work runs
            sent = 0
            while not future.done():
                await asyncio.sleep(0.25)
                while sent < len(progress_msgs):
                    yield _sse("progress", {"message": progress_msgs[sent]})
                    sent += 1

            # Drain any remaining progress after completion
            while sent < len(progress_msgs):
                yield _sse("progress", {"message": progress_msgs[sent]})
                sent += 1

            assessment = await future

            result = _assessment_response(assessment)
            yield _sse("result", {"data": result.model_dump()})

        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Recalculate (drop high-risk sites) ──────────────────────────────────────

class RecalculateRequest(BaseModel):
    assessment: dict
    threshold: int  # drop sites with risk_score > this value (30 or 60)


@app.post("/api/recalculate")
async def recalculate(body: RecalculateRequest):
    """
    Drop all sites whose legacy risk score exceeds `threshold`, then re-run
    the full scoring pipeline on the trimmed record set.
    """
    agent = _get_agent()
    threshold = body.threshold
    data = body.assessment

    # Identify site names to KEEP (score <= threshold)
    kept_names = {
        s["name"].upper().strip()
        for s in data.get("site_scores", [])
        if s.get("score", 0.0) <= threshold
    }
    if not kept_names:
        raise HTTPException(
            status_code=400,
            detail=f"All sites score above {threshold}; nothing would remain after dropping.",
        )

    from src.models.manufacturer import Manufacturer
    from src.models.osha_record import OSHARecord, Violation, AccidentSummary
    import datetime

    def _violations(raw: list) -> list:
        return [
            Violation(
                category=v.get("category", ""),
                severity=v.get("severity", ""),
                penalty_amount=v.get("penalty_amount", 0.0),
                is_repeat=v.get("is_repeat", False),
                is_willful=v.get("is_willful", False),
                description=v.get("description"),
                gravity=v.get("gravity"),
                nr_exposed=v.get("nr_exposed"),
                citation_id=v.get("citation_id"),
                gen_duty_narrative=v.get("gen_duty_narrative"),
            )
            for v in raw
        ]

    def _accidents(raw: list) -> list:
        return [
            AccidentSummary(
                summary_nr=a.get("summary_nr", ""),
                event_date=a.get("event_date"),
                event_desc=a.get("event_desc", ""),
                fatality=a.get("fatality", False),
                injuries=a.get("injuries", []),
                abstract=a.get("abstract", ""),
            )
            for a in raw
        ]

    def _parse_date(s: str) -> "datetime.date":
        try:
            return datetime.date.fromisoformat(s)
        except Exception:
            return datetime.date.today()

    all_records = [
        OSHARecord(
            inspection_id=r.get("inspection_id", ""),
            date_opened=_parse_date(r.get("date_opened", "")),
            violations=_violations(r.get("violations", [])),
            total_penalties=r.get("total_penalties", 0.0),
            severe_injury_or_fatality=r.get("severe_injury_or_fatality", False),
            accidents=_accidents(r.get("accidents", [])),
            naics_code=r.get("naics_code"),
            nr_in_estab=r.get("nr_in_estab"),
            estab_name=r.get("estab_name"),
            site_city=r.get("site_city"),
            site_state=r.get("site_state"),
        )
        for r in data.get("records", [])
    ]

    filtered = [
        r for r in all_records
        if (r.estab_name or "UNKNOWN").upper().strip() in kept_names
    ]
    if not filtered:
        raise HTTPException(
            status_code=400,
            detail="No records remain after filtering to kept sites.",
        )

    manufacturer = Manufacturer(name=data.get("manufacturer_name", ""))

    import asyncio
    import concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    assessment = await loop.run_in_executor(
        executor,
        lambda: agent.reassess(manufacturer, filtered),
    )
    return _assessment_response(assessment)


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    assessment: dict  # full AssessmentResponse payload from the client


@app.post("/api/chat")
async def chat(body: ChatRequest):
    """
    Answer a question about a completed assessment using the LLM.
    Reconstructs just enough of the assessment to call discuss_assessment.
    """
    agent = _get_agent()
    if not agent.client:
        return {"answer": "LLM features are not enabled — set GOOGLE_API_KEY to use Q&A."}

    # Rebuild a lightweight assessment stub the agent can use for Q&A
    from src.models.manufacturer import Manufacturer
    from src.models.assessment import RiskAssessment, ProbabilisticRiskTargets
    from src.models.osha_record import OSHARecord, Violation, AccidentSummary
    import datetime

    data = body.assessment

    def _rebuild_risk_targets(rt_data: dict | None):
        if not rt_data:
            return None
        try:
            return ProbabilisticRiskTargets(
                p_serious_wr_event=rt_data.get("p_serious_wr_event", 0.0),
                expected_penalty_usd_12m=rt_data.get("expected_penalty_usd_12m", 0.0),
                p_injury_event=rt_data.get("p_injury_event", 0.0),
                gravity_score=rt_data.get("gravity_score", 0.0),
                composite_risk_score=rt_data.get("composite_risk_score", 0.0),
            )
        except Exception:
            return None

    def _rebuild_violations(raw: list) -> list:
        out = []
        for v in raw:
            out.append(Violation(
                category=v.get("category", ""),
                severity=v.get("severity", ""),
                penalty_amount=v.get("penalty_amount", 0.0),
                is_repeat=v.get("is_repeat", False),
                is_willful=v.get("is_willful", False),
                description=v.get("description"),
                gravity=v.get("gravity"),
                nr_exposed=v.get("nr_exposed"),
                citation_id=v.get("citation_id"),
                gen_duty_narrative=v.get("gen_duty_narrative"),
            ))
        return out

    def _rebuild_accidents(raw: list) -> list:
        out = []
        for a in raw:
            out.append(AccidentSummary(
                summary_nr=a.get("summary_nr", ""),
                event_date=a.get("event_date"),
                event_desc=a.get("event_desc", ""),
                fatality=a.get("fatality", False),
                injuries=a.get("injuries", []),
                abstract=a.get("abstract", ""),
            ))
        return out

    records = []
    for r in data.get("records", []):
        try:
            date_opened = datetime.date.fromisoformat(r["date_opened"])
        except Exception:
            date_opened = datetime.date.today()
        records.append(OSHARecord(
            inspection_id=r.get("inspection_id", ""),
            date_opened=date_opened,
            violations=_rebuild_violations(r.get("violations", [])),
            total_penalties=r.get("total_penalties", 0.0),
            severe_injury_or_fatality=r.get("severe_injury_or_fatality", False),
            accidents=_rebuild_accidents(r.get("accidents", [])),
            naics_code=r.get("naics_code"),
            nr_in_estab=r.get("nr_in_estab"),
            estab_name=r.get("estab_name"),
            site_city=r.get("site_city"),
            site_state=r.get("site_state"),
        ))

    assessment = RiskAssessment(
        manufacturer=Manufacturer(name=data.get("manufacturer_name", "")),
        records=records,
        risk_score=data.get("risk_score", 0.0),
        recommendation=data.get("recommendation", ""),
        explanation=data.get("explanation", ""),
        confidence_score=data.get("confidence_score", 0.0),
        feature_weights=data.get("feature_weights", {}),
        percentile_rank=data.get("percentile_rank", 50.0),
        industry_label=data.get("industry_label", ""),
        industry_group=data.get("industry_group", ""),
        industry_percentile=data.get("industry_percentile", 50.0),
        industry_comparison=data.get("industry_comparison", []),
        missing_naics=data.get("missing_naics", False),
        establishment_count=data.get("establishment_count", 1),
        site_scores=data.get("site_scores", []),
        risk_concentration=data.get("risk_concentration", 0.0),
        systemic_risk_flag=data.get("systemic_risk_flag", False),
        aggregation_warning=data.get("aggregation_warning", ""),
        concentration_warning=data.get("concentration_warning", ""),
        risk_targets=_rebuild_risk_targets(data.get("risk_targets")),
    )

    import asyncio
    import concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    answer = await loop.run_in_executor(
        executor,
        lambda: agent.discuss_assessment(assessment, body.question),
    )
    return {"answer": answer}
