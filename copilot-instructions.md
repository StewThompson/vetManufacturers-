**Project Title:** Manufacturer Compliance Intelligence & Vetting Agent

**Project Summary**

This project aims to build an AI-assisted manufacturer risk prediction and vetting system that evaluates the future compliance and safety risk of individual manufacturers using publicly available enforcement, incident, and certification data. The system is intended to move beyond static summaries of past violations and toward a more predictive, explainable assessment of which manufacturers are more likely to experience serious compliance issues going forward.

The initial scope of the project is intentionally narrow and focused: **predicting and explaining risk for individual manufacturers**. The system will accept a manufacturer name (with optional location metadata), retrieve relevant public safety and compliance history, construct structured temporal features, and generate an explainable predictive risk profile. Rather than focusing on sourcing parts or procurement workflows, the current program is centered on identifying patterns that signal elevated future compliance risk and helping users understand the evidence behind those predictions.

---

**Phase 1 Goal — Individual Manufacturer Risk Prediction**

The system should:

1. Accept a manufacturer identity input (name + optional location, address, or domain).
2. Resolve and normalize the manufacturer into one or more physical establishments or facilities.
3. Retrieve publicly available enforcement and incident history, including:

   * workplace safety inspections
   * citations and violation categories
   * penalties and enforcement outcomes
   * severe injury and fatality signals where available
4. Extract structured features such as:

   * number and severity of violations
   * repeat or willful violation indicators
   * recency of inspections
   * penalty magnitude trends
   * recurring hazard themes
   * inspection frequency and violation density
5. Generate:

   * a predictive risk score
   * a recommendation or risk tier based on expected future compliance risk
   * a concise explanation highlighting the primary drivers of the prediction
6. Provide an interactive question-answer layer allowing users to ask:

   * what caused the risk score
   * whether violations are recent or systemic
   * whether repeat hazards exist
   * how risk has changed over time
   * whether the score reflects future risk or past history

The system must emphasize **explainability, temporal correctness, and evidence-backed reasoning**. Absence of enforcement records must not be interpreted as proof of safety, and sparse data should be surfaced as uncertainty rather than treated as low risk.

---

**Architecture Principles**

* Deterministic data retrieval and temporal feature construction form the foundation.
* The AI agent operates as an analysis and interaction layer, not as the primary data collector.
* Entity resolution confidence must be tracked and surfaced when ambiguity exists.
* Predictive models must be trained on real future outcomes rather than heuristic pseudo-labels.
* All recommendations and risk outputs must cite supporting evidence (dates, inspection identifiers, violation categories).
* Never change file content with console commands; just edit them with copilot tools or report to user if theres an issue with current tools. 

---

**Expansion Roadmap (Future Phases)**

**Phase 2 — Multi-source Compliance Intelligence**

* Add environmental enforcement history
* Incorporate certification signals (quality and process standards)
* Improve parent/subsidiary and multi-facility resolution

**Phase 3 — Comparative Risk Intelligence**

* Benchmark manufacturers against industry peers
* Add percentile-based and cohort-based risk comparisons
* Surface trends in risk over time across manufacturers and industries

**Phase 4 — Monitoring and Decision Support**

* supplier monitoring and alerting over time
* historical versus predicted risk tracking
* integration into broader operational or sourcing decisions if needed

---

**Agent Responsibilities**

The AI agent should be able to:

* interpret ambiguous manufacturer inputs
* summarize enforcement history into predictive and actionable insight
* answer follow-up questions using structured evidence
* highlight uncertainty, missing data, and potential false confidence
* distinguish between historical severity and future predicted risk
* avoid unsupported conclusions when records are sparse

The agent must behave as a compliance analyst and risk advisor rather than a generic conversational assistant.

---

**Success Criteria for Phase 1**

* Reliable manufacturer identity resolution
* Accurate retrieval and structuring of enforcement history
* Transparent, explainable predictive risk scoring
* Useful conversational exploration of incident history and future risk
* Clear communication of limitations, uncertainty, and temporal assumptions

---

**Long-Term Vision**

The project evolves into a manufacturer compliance intelligence layer that helps users understand, compare, and monitor safety and enforcement risk across industrial suppliers. The long-term goal is to provide a structured and explainable way to forecast compliance risk before it materializes, enabling better decisions grounded in public enforcement history, predictive modeling, and transparent evidence.