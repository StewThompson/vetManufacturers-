// ── Search types ─────────────────────────────────────────────────────────────

export interface FacilityOut {
  raw_name: string
  display_name: string
  facility_code: string | null
  city: string
  state: string
  address: string
  naics_code: string
  confidence: number
  confidence_label: string
}

export interface GroupedCompanyOut {
  parent_name: string
  total_facilities: number
  confidence: number
  confidence_label: string
  high_confidence: FacilityOut[]
  medium_confidence: FacilityOut[]
  low_confidence: FacilityOut[]
}

export interface SearchResponse {
  query: string
  top_group: GroupedCompanyOut | null
  other_groups: GroupedCompanyOut[]
  unmatched: string[]
}

// ── Assessment types ─────────────────────────────────────────────────────────

export interface ViolationOut {
  category: string
  severity: string
  penalty_amount: number
  is_repeat: boolean
  is_willful: boolean
  description: string | null
  gravity: string | null
  nr_exposed: number | null
  citation_id: string | null
  gen_duty_narrative: string | null
}

export interface AccidentOut {
  summary_nr: string
  event_date: string | null
  event_desc: string
  fatality: boolean
  injuries: Record<string, unknown>[]
  abstract: string
}

export interface OSHARecordOut {
  inspection_id: string
  date_opened: string
  violations: ViolationOut[]
  total_penalties: number
  severe_injury_or_fatality: boolean
  accidents: AccidentOut[]
  naics_code: string | null
  nr_in_estab: string | null
  estab_name: string | null
  site_city: string | null
  site_state: string | null
}

export interface SiteScoreOut {
  name: string
  score: number
  n_inspections: number
  naics_code: string | null
  city: string | null
  state: string | null
}

export type Recommendation = 'Recommend' | 'Proceed with Caution' | 'Do Not Recommend'

export interface ComplianceOutlook12M {
  expected_inspections_12m: number
  expected_violations_12m: number
  expected_penalties_usd_12m: number
  expected_serious_12m: number
  expected_willful_repeat_12m: number
  risk_band: 'low' | 'moderate' | 'high'
  has_history: boolean
  basis: string
  summary_narrative: string
}

export interface ProbabilisticRiskTargetsOut {
  /** Head 1: probability of any Serious/Willful/Repeat citation within 12 months */
  p_serious_wr_event: number
  /** Head 2: expected total OSHA penalty (USD) */
  expected_penalty_usd_12m: number
  /** Head 3: expected total violations/citations */
  expected_citations_12m: number
  /** Head 4: probability of exceeding NAICS-adjusted P75 penalty threshold */
  p_moderate_penalty_event: number
  /** Head 4: probability of exceeding NAICS-adjusted P90 penalty threshold */
  p_large_penalty_event: number
  /** Head 4: probability of exceeding NAICS-adjusted P95 penalty threshold */
  p_extreme_penalty_event: number
  /** Weighted composite of all four heads (0–100) */
  composite_risk_score: number
  /** NAICS-adjusted P90 dollar threshold used for large-penalty head */
  large_penalty_threshold_usd: number
}

export interface AssessmentResponse {
  manufacturer_name: string
  risk_score: number
  recommendation: Recommendation
  explanation: string
  confidence_score: number
  feature_weights: Record<string, number>
  percentile_rank: number
  industry_label: string
  industry_group: string
  industry_percentile: number
  industry_comparison: string[]
  missing_naics: boolean
  establishment_count: number
  site_scores: SiteScoreOut[]
  risk_concentration: number
  systemic_risk_flag: boolean
  aggregation_warning: string
  concentration_warning: string
  records: OSHARecordOut[]
  record_count: number
  outlook: ComplianceOutlook12M | null
  risk_targets: ProbabilisticRiskTargetsOut | null
}

// ── SSE event types ───────────────────────────────────────────────────────────

export interface SSEProgressEvent {
  type: 'progress'
  message: string
}

export interface SSEResultEvent {
  type: 'result'
  data: AssessmentResponse
}

export interface SSEErrorEvent {
  type: 'error'
  message: string
}

export type SSEEvent = SSEProgressEvent | SSEResultEvent | SSEErrorEvent

// ── Assess request params ─────────────────────────────────────────────────────

export interface AssessParams {
  raw_names?: string[]    // specific facility OSHA names
  company?: string        // or search by company name
  display_name?: string
  years_back?: number
}
