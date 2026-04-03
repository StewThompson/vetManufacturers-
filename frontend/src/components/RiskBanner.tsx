import type { AssessmentResponse, Recommendation } from '../types/assessment'

const REC_COLORS: Record<Recommendation, string> = {
  'Recommend': 'green',
  'Proceed with Caution': 'yellow',
  'Do Not Recommend': 'red',
}

const SCORE_COLORS: Record<'green' | 'yellow' | 'red', { stroke: string; trackStroke: string }> = {
  green:  { stroke: '#059669', trackStroke: '#d1fae5' },
  yellow: { stroke: '#d97706', trackStroke: '#fef3c7' },
  red:    { stroke: '#dc2626', trackStroke: '#fee2e2' },
}

function ScoreRing({ score, color }: { score: number; color: 'green' | 'yellow' | 'red' }) {
  const R = 48, SW = 9, CX = 60, CY = 60, SIZE = 120
  const circumference = 2 * Math.PI * R
  const fillArc = Math.min(score / 100, 1) * circumference
  const gap = circumference - fillArc
  const { stroke, trackStroke } = SCORE_COLORS[color]

  return (
    <div style={{ position: 'relative', width: SIZE, height: SIZE, flexShrink: 0 }}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ display: 'block' }}>
        <defs>
          <filter id={`glow-${color}`} x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <circle cx={CX} cy={CY} r={R} fill="none" stroke={trackStroke} strokeWidth={SW} />
        {score > 0 && (
          <circle
            cx={CX} cy={CY} r={R} fill="none"
            stroke={stroke} strokeWidth={SW}
            strokeDasharray={`${fillArc} ${gap}`}
            strokeLinecap="round"
            transform={`rotate(-90 ${CX} ${CY})`}
            filter={`url(#glow-${color})`}
          />
        )}
        <text
          x={CX} y={CY - 5}
          textAnchor="middle" dominantBaseline="middle"
          style={{ fontSize: 22, fontWeight: 800, fill: stroke, fontFamily: 'inherit' }}
        >
          {score.toFixed(0)}
        </text>
        <text
          x={CX} y={CY + 13}
          textAnchor="middle"
          style={{ fontSize: 10, fill: '#94a3b8', fontFamily: 'inherit' }}
        >
          / 100
        </text>
      </svg>
      <div
        style={{
          position: 'absolute',
          bottom: 4,
          left: '50%',
          transform: 'translateX(-50%)',
          fontSize: 8,
          fontWeight: 700,
          background: 'var(--accent-dim)',
          color: 'var(--accent)',
          border: '1px solid var(--accent)',
          borderRadius: 3,
          padding: '1px 5px',
          letterSpacing: '0.4px',
          textTransform: 'uppercase',
          whiteSpace: 'nowrap',
        }}
      >
        ML Composite
      </div>
    </div>
  )
}

export function RiskBanner({ result }: { result: AssessmentResponse }) {
  const color = REC_COLORS[result.recommendation] as 'green' | 'yellow' | 'red'
  const percentile = result.missing_naics ? result.percentile_rank : result.industry_percentile

  return (
    <div className={`risk-banner risk-banner-${color}`}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <ScoreRing score={result.risk_score} color={color} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div>
            <div className="risk-score-label">{result.manufacturer_name}</div>
            <div className="risk-score-label" style={{ marginTop: 2 }}>
              {result.industry_label}
            </div>
          </div>
          <div>
            <div className="risk-score-label">
              {result.missing_naics ? 'Pop. Percentile' : 'Industry Percentile'}
            </div>
            <div className={`risk-score-value risk-score-${color}`} style={{ fontSize: 28, lineHeight: 1.1 }}>
              {percentile.toFixed(0)}<sup style={{ fontSize: 14 }}>th</sup>
            </div>
          </div>
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div
          className={`badge badge-${color}`}
          style={{ fontSize: 14, padding: '6px 14px' }}
        >
          {result.recommendation}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
          {result.establishment_count} {result.establishment_count === 1 ? 'site' : 'sites'}
        </div>
      </div>
    </div>
  )
}
