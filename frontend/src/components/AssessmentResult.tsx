import type { AssessmentResponse, ComplianceOutlook12M, ProbabilisticRiskTargetsOut, Recommendation } from '../types/assessment'

const REC_COLORS: Record<Recommendation, string> = {
  'Recommend': 'green',
  'Proceed with Caution': 'yellow',
  'Do Not Recommend': 'red',
}

interface Props {
  messages: string[]
  isRunning: boolean
  result: AssessmentResponse | null
  error: string | null
  activeTab: string
  onTabChange: (tab: string) => void
}

export default function ProgressStream({
  messages,
  isRunning,
  result,
  error,
}: Pick<Props, 'messages' | 'isRunning' | 'result' | 'error'>) {
  if (!isRunning && !result && !error && messages.length === 0) return null

  return (
    <div className="card">
      <div className="card-title">Progress</div>
      <div className="progress-list">
        {messages.map((msg, i) => {
          const isCurrent = isRunning && i === messages.length - 1
          return (
            <div key={i} className={`progress-item ${isCurrent ? 'current' : ''}`}>
              {isCurrent ? (
                <div className="spinner" />
              ) : (
                <div className="progress-dot" style={{ background: 'var(--success)' }} />
              )}
              <span>{msg}</span>
            </div>
          )
        })}
        {isRunning && messages.length === 0 && (
          <div className="progress-item current">
            <div className="spinner" />
            <span>Initialising…</span>
          </div>
        )}
      </div>
      {error && (
        <div style={{ color: 'var(--danger)', fontSize: 13, marginTop: 12 }}>
          ⚠ {error}
        </div>
      )}
      {!isRunning && result && (
        <div
          style={{
            marginTop: 12,
            fontSize: 12,
            color: 'var(--success)',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <div className="progress-dot" />
          Assessment complete
        </div>
      )}
    </div>
  )
}

// ── Risk banner ───────────────────────────────────────────────────────────────

export function RiskBanner({ result }: { result: AssessmentResponse }) {
  const color = REC_COLORS[result.recommendation]
  return (
    <div className={`risk-banner risk-banner-${color}`}>
      <div>
        <div className="risk-score-label">{result.manufacturer_name}</div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, marginTop: 4 }}>
          <div>
            <div className="risk-score-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              Risk Score
              {result.risk_targets && (
                <span
                  style={{
                    fontSize: 9,
                    fontWeight: 600,
                    background: 'var(--accent-dim)',
                    color: 'var(--accent)',
                    border: '1px solid var(--accent)',
                    borderRadius: 3,
                    padding: '1px 5px',
                    letterSpacing: '0.3px',
                    textTransform: 'uppercase',
                  }}
                >
                  ML Composite
                </span>
              )}
            </div>
            <div className={`risk-score-value risk-score-${color}`}>
              {result.risk_score.toFixed(1)}
              <span style={{ fontSize: 16, fontWeight: 400, opacity: 0.7 }}>&thinsp;/ 100</span>
            </div>
          </div>
          <div>
            <div className="risk-score-label">{result.missing_naics ? 'Pop. Percentile' : 'Industry Percentile'}</div>
            <div className={`risk-score-value risk-score-${color}`} style={{ fontSize: 28 }}>
              {(result.missing_naics ? result.percentile_rank : result.industry_percentile).toFixed(0)}
              <sup style={{ fontSize: 14 }}>th</sup>
            </div>
          </div>
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div
          className={`badge badge-${color === 'green' ? 'green' : color === 'yellow' ? 'yellow' : 'red'}`}
          style={{ fontSize: 14, padding: '6px 14px' }}
        >
          {result.recommendation}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
          {result.industry_label} · {result.establishment_count} sites
        </div>
      </div>
    </div>
  )
}

// ── Stats grid ────────────────────────────────────────────────────────────────

export function StatsGrid({ result }: { result: AssessmentResponse }) {
  const totalPenalties = result.records.reduce((s, r) => s + r.total_penalties, 0)
  const totalViolations = result.records.reduce((s, r) => s + r.violations.length, 0)
  const fatalities = result.records.filter((r) => r.severe_injury_or_fatality).length
  const willfulCount = result.records.flatMap((r) => r.violations).filter((v) => v.is_willful).length
  const repeatCount = result.records.flatMap((r) => r.violations).filter((v) => v.is_repeat).length

  return (
    <div className="stat-grid">
      <Stat label="Total Inspections" value={result.record_count} />
      <Stat label="Total Violations" value={totalViolations} />
      <Stat
        label="Total Penalties"
        value={`$${totalPenalties.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
        color="var(--warning)"
      />
      <Stat label="Serious/Fatal" value={fatalities} color={fatalities > 0 ? 'var(--danger)' : undefined} />
      <Stat label="Willful" value={willfulCount} color={willfulCount > 0 ? 'var(--danger)' : undefined} />
      <Stat label="Repeat" value={repeatCount} color={repeatCount > 0 ? 'var(--warning)' : undefined} />
      <Stat label="Sites Assessed" value={result.establishment_count} />
      <Stat label="Risk Concentration" value={`${(result.risk_concentration * 100).toFixed(0)}%`} />
    </div>
  )
}

function Stat({
  label,
  value,
  color,
}: {
  label: string
  value: string | number
  color?: string
}) {
  return (
    <div className="stat-item">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>
        {value}
      </div>
    </div>
  )
}

// ── Explanation ───────────────────────────────────────────────────────────────

export function ExplanationPanel({ result }: { result: AssessmentResponse }) {
  const maxWeight = Math.max(...Object.values(result.feature_weights), 0.01)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {result.aggregation_warning && (
        <div className="warning-box">⚠ {result.aggregation_warning}</div>
      )}
      {result.concentration_warning && (
        <div className="warning-box">⚠ {result.concentration_warning}</div>
      )}
      {result.systemic_risk_flag && result.risk_score >= 45 && (
        <div className="warning-box" style={{ background: 'var(--danger-bg)', borderColor: 'var(--danger-border)', color: 'var(--danger)' }}>
          ⛔ Systemic risk detected across multiple sites
        </div>
      )}

      <div className="card">
        <div className="card-title">Assessment Summary</div>
        <p className="explanation-text">{result.explanation}</p>
      </div>

      <div className="card">
        <div className="card-title">Industry Context</div>
        <p className="explanation-text">
          {result.industry_group} · {result.industry_label}
        </p>
        {result.industry_comparison.map((line, i) => (
          <p key={i} className="explanation-text" style={{ marginTop: 6 }}>
            {line}
          </p>
        ))}
      </div>

      <div className="card">
        <div className="card-title">Model Feature Weights</div>
        {Object.entries(result.feature_weights)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 15)
          .map(([key, val]) => (
            <div key={key} className="feature-row">
              <div className="feature-name">{key.replace(/_/g, ' ')}</div>
              <div className="feature-bar-bg">
                <div
                  className="feature-bar-fill"
                  style={{ width: `${(val / maxWeight) * 100}%` }}
                />
              </div>
              <div className="feature-value">{val.toFixed(3)}</div>
            </div>
          ))}
      </div>
    </div>
  )
}

// ── 12-month Compliance Outlook ───────────────────────────────────────────────

export function OutlookPanel({ outlook }: { outlook: ComplianceOutlook12M }) {
  const bandColor = outlook.risk_band === 'high'
    ? 'var(--danger)'
    : outlook.risk_band === 'moderate'
    ? 'var(--warning)'
    : 'var(--success)'

  const bandLabel = outlook.risk_band === 'high'
    ? 'High Risk'
    : outlook.risk_band === 'moderate'
    ? 'Moderate Risk'
    : 'Low Risk'

  return (
    <div className="card">
      <div className="card-title">
        12-Month Compliance Outlook
        <span
          style={{
            marginLeft: 10,
            fontSize: 11,
            fontWeight: 600,
            padding: '2px 8px',
            borderRadius: 4,
            background: `color-mix(in srgb, ${bandColor} 15%, transparent)`,
            color: bandColor,
            border: `1px solid color-mix(in srgb, ${bandColor} 30%, transparent)`,
          }}
        >
          {bandLabel}
        </span>
      </div>

      <p className="explanation-text" style={{ marginBottom: 16 }}>
        {outlook.summary_narrative}
      </p>

      <div className="stat-grid">
        <OutlookStat
          label="Expected Inspections"
          value={outlook.expected_inspections_12m.toFixed(1)}
          sub="visits / 12 mo"
        />
        <OutlookStat
          label="Expected Violations"
          value={outlook.expected_violations_12m.toFixed(1)}
          sub="total violations"
          color={outlook.expected_violations_12m >= 2 ? 'var(--warning)' : undefined}
        />
        <OutlookStat
          label="Expected Penalties"
          value={`$${outlook.expected_penalties_usd_12m.toLocaleString('en-US')}`}
          sub="estimated OSHA fines"
          color={outlook.expected_penalties_usd_12m >= 5000 ? 'var(--warning)' : undefined}
        />
        <OutlookStat
          label="Serious Violations"
          value={outlook.expected_serious_12m.toFixed(1)}
          sub="Serious-type citations"
          color={outlook.expected_serious_12m >= 1 ? 'var(--danger)' : undefined}
        />
        <OutlookStat
          label="Willful / Repeat"
          value={outlook.expected_willful_repeat_12m.toFixed(1)}
          sub="Willful + Repeat citations"
          color={outlook.expected_willful_repeat_12m >= 0.5 ? 'var(--danger)' : undefined}
        />
      </div>

      <div
        style={{
          marginTop: 12,
          fontSize: 11,
          color: 'var(--text-muted)',
          borderTop: '1px solid var(--border)',
          paddingTop: 8,
        }}
      >
        Basis: {outlook.basis}
      </div>
    </div>
  )
}

function OutlookStat({
  label,
  value,
  sub,
  color,
}: {
  label: string
  value: string
  sub: string
  color?: string
}) {
  return (
    <div className="stat-item">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>
        {value}
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>
    </div>
  )
}

// ── Multi-Target Probabilistic Risk Panel ─────────────────────────────────────

export function RiskTargetsPanel({ targets }: { targets: ProbabilisticRiskTargetsOut }) {
  return (
    <div className="card">
      <div
        className="card-title"
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
      >
        Predictive Risk Breakdown
        <span
          style={{
            fontSize: 11,
            color: 'var(--text-muted)',
            fontWeight: 400,
            background: 'var(--surface-2)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: '2px 8px',
          }}
        >
          Multi-Target ML · 4 Heads
        </span>
      </div>

      {/* Probability bars — 2-column grid */}
      <div
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 32px', marginBottom: 16 }}
      >
        <div>
          <ProbabilityBar
            label="Serious / Willful / Repeat Event"
            prob={targets.p_serious_wr_event}
            thresholds={[0.25, 0.5]}
            sub="Head 1 · probability of any S/WR citation within 12 mo"
          />
          <ProbabilityBar
            label="Moderate Penalty (P75)"
            prob={targets.p_moderate_penalty_event}
            thresholds={[0.3, 0.5]}
            sub="Head 4 · industry-adjusted ≥ NAICS P75 threshold"
          />
        </div>
        <div>
          <ProbabilityBar
            label="Large Penalty (P90)"
            prob={targets.p_large_penalty_event}
            thresholds={[0.2, 0.4]}
            sub={`Head 4 · threshold ≥ $${targets.large_penalty_threshold_usd.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          />
          <ProbabilityBar
            label="Extreme Penalty (P95)"
            prob={targets.p_extreme_penalty_event}
            thresholds={[0.1, 0.25]}
            sub="Head 4 · industry-adjusted ≥ NAICS P95 threshold"
          />
        </div>
      </div>

      {/* Regression head stats */}
      <div className="stat-grid" style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
        <OutlookStat
          label="Expected Penalty"
          value={`$${targets.expected_penalty_usd_12m.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          sub="Head 2 · 12-month total (USD)"
          color={
            targets.expected_penalty_usd_12m >= 10_000
              ? 'var(--danger)'
              : targets.expected_penalty_usd_12m >= 2_000
              ? 'var(--warning)'
              : undefined
          }
        />
        <OutlookStat
          label="Expected Citations"
          value={targets.expected_citations_12m.toFixed(1)}
          sub="Head 3 · 12-month total"
          color={
            targets.expected_citations_12m >= 5
              ? 'var(--danger)'
              : targets.expected_citations_12m >= 2
              ? 'var(--warning)'
              : undefined
          }
        />
      </div>
    </div>
  )
}

function ProbabilityBar({
  label,
  prob,
  thresholds,
  sub,
}: {
  label: string
  prob: number
  /** [warn_threshold, danger_threshold] */
  thresholds: [number, number]
  sub?: string
}) {
  const pct = Math.min(prob * 100, 100)
  const barColor =
    prob >= thresholds[1]
      ? 'var(--danger)'
      : prob >= thresholds[0]
      ? 'var(--warning)'
      : 'var(--success)'

  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 12,
          marginBottom: 4,
        }}
      >
        <span style={{ color: 'var(--text-muted)' }}>{label}</span>
        <span style={{ fontWeight: 600, color: barColor }}>{pct.toFixed(0)}%</span>
      </div>
      <div
        style={{
          background: 'var(--surface-2)',
          border: '1px solid var(--border)',
          borderRadius: 4,
          height: 8,
          overflow: 'hidden',
        }}
      >
        <div
          style={{ width: `${pct}%`, height: '100%', background: barColor, borderRadius: 4 }}
        />
      </div>
      {sub && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>
      )}
    </div>
  )
}
