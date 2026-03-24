import type { AssessmentResponse, Recommendation } from '../types/assessment'

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
            <div className="risk-score-label">Risk Score</div>
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
