import type { AssessmentResponse, ComplianceOutlook12M, ProbabilisticRiskTargetsOut } from '../types/assessment'

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

function Stat({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="stat-item">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>
        {value}
      </div>
    </div>
  )
}

export function OutlookPanel({ outlook }: { outlook: ComplianceOutlook12M }) {
  const bandColor =
    outlook.risk_band === 'high'
      ? 'var(--danger)'
      : outlook.risk_band === 'moderate'
      ? 'var(--warning)'
      : 'var(--success)'

  const bandLabel =
    outlook.risk_band === 'high' ? 'High Risk' : outlook.risk_band === 'moderate' ? 'Moderate Risk' : 'Low Risk'

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
        <OutlookStat label="Expected Inspections" value={outlook.expected_inspections_12m.toFixed(1)} sub="visits / 12 mo" />
        <OutlookStat
          label="Expected Violations"
          value={outlook.expected_violations_12m.toFixed(1)}
          sub="total violations"
          color={outlook.expected_violations_12m >= 2 ? 'var(--warning)' : undefined}
        />
        <OutlookStat
          label="Large Penalty"
          value={`$${outlook.expected_penalties_usd_12m.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          sub="industry P90 threshold"
          color={undefined}
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

export function RiskTargetsPanel({ targets }: { targets: ProbabilisticRiskTargetsOut }) {
  const gravityLevel =
    targets.gravity_score >= 200
      ? { label: 'High', color: 'var(--danger)' }
      : targets.gravity_score >= 60
      ? { label: 'Moderate', color: 'var(--warning)' }
      : { label: 'Low', color: 'var(--success)' }

  const penaltyLevel =
    targets.expected_penalty_usd_12m >= 100_000
      ? { label: 'High', color: 'var(--danger)' }
      : targets.expected_penalty_usd_12m >= 25_000
      ? { label: 'Moderate', color: 'var(--warning)' }
      : { label: 'Low', color: 'var(--success)' }

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
          Multi-Target ML · 5 Heads
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 32px', marginBottom: 16 }}>
        <div>
          <ProbabilityBar
            label="Serious / Willful / Repeat Event"
            prob={targets.p_serious_wr_event}
            thresholds={[0.25, 0.5]}
            sub="Head 1 · probability of any S/WR citation within 12 mo"
          />
          <ProbabilityBar
            label="Hospitalization / Fatality"
            prob={targets.p_injury_event}
            thresholds={[0.05, 0.15]}
            sub="Head 3 · probability of hospitalized or fatal injury"
          />
        </div>
        <div>
          <ProbabilityBar
            label="Probability of Large Penalty"
            prob={targets.p_penalty_ge_p90}
            thresholds={[0.05, 0.15]}
            sub={`Head 2 · probability of penalty ≥ industry P90`}
          />
          <ProbabilityBar
            label="Probability of Extreme Penalty"
            prob={targets.p_penalty_ge_p95}
            thresholds={[0.03, 0.10]}
            sub={`Head 2b · probability of penalty ≥ industry P95`}
          />
        </div>
      </div>

      {/* Legacy regression heads — now active contributors to composite score */}
      <div
        style={{
          borderTop: '1px solid var(--border)',
          paddingTop: 12,
          marginTop: 4,
        }}
      >
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, fontWeight: 600 }}>
          Severity Signal Heads (active in composite)
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 32px' }}>
          <LegacyMetric
            label="Gravity-Weighted Severity"
            value={targets.gravity_score.toFixed(1)}
            level={gravityLevel}
            sub={`Head 4 · Σ(gravity × citation weight); composite input`}
          />
          <LegacyMetric
            label="Expected Penalty (12 mo)"
            value={`$${targets.expected_penalty_usd_12m.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
            level={penaltyLevel}
            sub={`Head 2a · hurdle model estimate; industry P90 = $${targets.industry_p90_penalty.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          />
        </div>
      </div>
    </div>
  )
}

function LegacyMetric({
  label,
  value,
  level,
  sub,
}: {
  label: string
  value: string
  level: { label: string; color: string }
  sub?: string
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: 'var(--text-muted)' }}>{label}</span>
        <span
          style={{
            fontWeight: 600,
            color: level.color,
            fontSize: 11,
            padding: '1px 6px',
            borderRadius: 3,
            background: `color-mix(in srgb, ${level.color} 12%, transparent)`,
            border: `1px solid color-mix(in srgb, ${level.color} 25%, transparent)`,
          }}
        >
          {level.label}
        </span>
      </div>
      <div style={{ fontWeight: 700, fontSize: 16, color: level.color }}>{value}</div>
      {sub && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>
      )}
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
    prob >= thresholds[1] ? 'var(--danger)' : prob >= thresholds[0] ? 'var(--warning)' : 'var(--success)'

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
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
        <div style={{ width: `${pct}%`, height: '100%', background: barColor, borderRadius: 4 }} />
      </div>
      {sub && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>
      )}
    </div>
  )
}
