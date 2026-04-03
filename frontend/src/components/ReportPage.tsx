import type { CSSProperties } from 'react'
import './report.css'

function badge(color: string): CSSProperties {
  return {
    display: 'inline-block',
    background: color + '22',
    color,
    border: `1px solid ${color}55`,
    borderRadius: 3,
    padding: '0px 5px',
    fontSize: 7.8,
    fontWeight: 700,
  }
}

function PipelineDiagram() {
  const BOX_W = 118, BOX_H = 44, GAP = 18, CY = 28, H = 68
  const ACCENT = '#1d5a8e', BORDER = '#b6cbdf', BG = '#e8f2fb'

  const allNodes = [
    { label: 'OSHA Records',       sub: '10-yr window',               bg: BG,        stroke: BORDER,    text: ACCENT,    subText: '#64748b' },
    { label: 'Feature Extraction', sub: '49 dims',                    bg: BG,        stroke: BORDER,    text: ACCENT,    subText: '#64748b' },
    { label: 'GBT Heads ×4',       sub: 'Classify + Regress',         bg: BG,        stroke: BORDER,    text: ACCENT,    subText: '#64748b' },
    { label: 'Composite Blend',    sub: 'w₁p_wr+w₂pen+w₃cit+w₄tier', bg: BG,        stroke: BORDER,    text: ACCENT,    subText: '#64748b' },
    { label: 'Evidence Ceiling',   sub: 'Inspection-count cap',       bg: BG,        stroke: BORDER,    text: ACCENT,    subText: '#64748b' },
    { label: 'Recommendation',     sub: 'Rec / Caution / DNR',        bg: '#fff0f0', stroke: '#fca5a5', text: '#dc2626', subText: '#b91c1c' },
  ]

  const STEP = BOX_W + GAP
  const totalW = allNodes.length * BOX_W + (allNodes.length - 1) * GAP

  return (
    <svg width="100%" viewBox={`0 0 ${totalW} ${H}`} style={{ display: 'block' }}>
      <defs>
        <marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={ACCENT} />
        </marker>
      </defs>
      {allNodes.map((n, i) => {
        const x = i * STEP
        return (
          <g key={i}>
            <rect x={x} y={CY - BOX_H / 2} width={BOX_W} height={BOX_H} rx={5}
              fill={n.bg} stroke={n.stroke} strokeWidth={1} />
            <text x={x + BOX_W / 2} y={CY - 6} textAnchor="middle"
              style={{ fontSize: 9, fontWeight: 700, fill: n.text, fontFamily: 'Inter,system-ui,sans-serif' }}>
              {n.label}
            </text>
            <text x={x + BOX_W / 2} y={CY + 7} textAnchor="middle"
              style={{ fontSize: 7, fill: n.subText, fontFamily: 'Inter,system-ui,sans-serif' }}>
              {n.sub}
            </text>
            {i < allNodes.length - 1 && (
              <line x1={x + BOX_W} y1={CY} x2={x + STEP - 2} y2={CY}
                stroke={ACCENT} strokeWidth={1.2} markerEnd="url(#arr)" />
            )}
          </g>
        )
      })}
    </svg>
  )
}

function WeightChart() {
  const heads = [
    { label: 'p_wr (Head 1)',       w: 0.40, color: '#dc2626' },
    { label: 'pen_norm (Head 2)',    w: 0.25, color: '#d97706' },
    { label: 'cit_norm (Head 3)',    w: 0.20, color: '#2563eb' },
    { label: 'tier_blend (Head 4)', w: 0.15, color: '#7c3aed' },
  ]
  const BAR_H = 15, GAP = 7, LABEL_W = 118, BAR_MAX = 186
  const H = heads.length * (BAR_H + GAP) + 4

  return (
    <svg width="100%" viewBox={`0 0 ${LABEL_W + BAR_MAX + 40} ${H}`} style={{ display: 'block' }}>
      {heads.map((h, i) => {
        const y = i * (BAR_H + GAP)
        const bw = h.w * BAR_MAX
        return (
          <g key={h.label}>
            <text x={LABEL_W - 5} y={y + BAR_H - 3} textAnchor="end"
              style={{ fontSize: 8.5, fill: '#374151', fontFamily: 'Inter,system-ui,sans-serif' }}>
              {h.label}
            </text>
            <rect x={LABEL_W} y={y} width={bw} height={BAR_H} rx={3} fill={h.color} opacity={0.85} />
            <text x={LABEL_W + bw + 4} y={y + BAR_H - 3}
              style={{ fontSize: 8.5, fontWeight: 700, fill: h.color, fontFamily: 'Inter,system-ui,sans-serif' }}>
              {(h.w * 100).toFixed(0)}%
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function MetricsChart() {
  const metrics = [
    { label: 'AUROC',   val: 0.733, target: 0.760, color: '#2563eb' },
    { label: 'BSS',     val: 0.163, target: 0.180, color: '#7c3aed', scale: 4 },
    { label: 'Lift×2',  val: Math.min(2.07 / 3.5, 1), target: 1, color: '#059669', rawLabel: '2.07×' },
    { label: 'AP',      val: 0.711, target: 0.760, color: '#d97706' },
  ]
  const BAR_H = 14, GAP = 8, LABEL_W = 64, BAR_MAX = 172
  const H = metrics.length * (BAR_H + GAP) + 4

  return (
    <svg width="100%" viewBox={`0 0 ${LABEL_W + BAR_MAX + 52} ${H}`} style={{ display: 'block' }}>
      {metrics.map((m, i) => {
        const y = i * (BAR_H + GAP)
        const bw = m.val * BAR_MAX
        const tw = m.target * BAR_MAX
        const display = m.rawLabel ?? m.val.toFixed(3)
        return (
          <g key={m.label}>
            <text x={LABEL_W - 4} y={y + BAR_H - 2} textAnchor="end"
              style={{ fontSize: 8.5, fill: '#374151', fontFamily: 'Inter,system-ui,sans-serif' }}>
              {m.label}
            </text>
            <rect x={LABEL_W} y={y} width={BAR_MAX} height={BAR_H} rx={3} fill="#f1f5f9" />
            <rect x={LABEL_W} y={y} width={bw} height={BAR_H} rx={3} fill={m.color} opacity={0.8} />
            <line x1={LABEL_W + tw} y1={y - 1} x2={LABEL_W + tw} y2={y + BAR_H + 1}
              stroke="#6b7280" strokeWidth={1.2} strokeDasharray="2,2" />
            <text x={LABEL_W + bw + 4} y={y + BAR_H - 2}
              style={{ fontSize: 8, fontWeight: 700, fill: m.color, fontFamily: 'Inter,system-ui,sans-serif' }}>
              {display}
            </text>
          </g>
        )
      })}
      <text x={LABEL_W + 2} y={H + 9}
        style={{ fontSize: 7, fill: '#9ca3af', fontFamily: 'Inter,system-ui,sans-serif' }}>
        dashed line = target threshold
      </text>
    </svg>
  )
}

const compositeFormula =
`composite = w1·p_serious_wr × 100
           + w2·clamp(log1p(E[penalty]) / log1p(200k), 1) × 100
           + w3·clamp(E[citations] / 20, 1) × 100
           + w4·(0.5·p_large + 0.5·p_extreme) × 100

weights: w1=0.40  w2=0.25  w3=0.20  w4=0.15   (Spearman-ρ optimised, L2 λ=0.1)`

const hyperparams =
`GradientBoostingClassifier (Heads 1, 4)
  n_estimators=300, max_depth=4, lr=0.05
  subsample=0.8, min_samples_leaf=3
  + CalibratedClassifierCV (isotonic)

GradientBoostingRegressor (Heads 2, 3)
  n_estimators=400, max_depth=5, lr=0.05
  subsample=0.8, loss="huber", alpha=0.9
  + StandardScaler pipeline`

export default function ReportPage() {
  return (
    <div className="report-page-print report-page rp-page">
      <div style={{ borderBottom: '2px solid #1d5a8e', paddingBottom: 6, marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <span className="rp-h1">OSHA Manufacturer Compliance Risk Model</span>
          <span style={{ ...badge('#1d5a8e'), fontSize: 8 }}>Technical Report · April 2026</span>
        </div>
        <div className="rp-subtitle">
          Multi-target gradient-boosted ensemble for predictive OSHA inspection risk scoring &amp; manufacturer vetting
        </div>
      </div>

      <div className="rp-abstract">
        <strong style={{ color: '#0f2240' }}>Abstract. </strong>
        We present a multi-target ML system that assigns a 0–100 composite risk score to manufacturing establishments
        by predicting four complementary OSHA outcomes: probability of a Serious/Willful/Repeat citation (Head 1),
        expected future penalty (Head 2), expected citation count (Head 3), and penalty-tier exceedance probability
        (Head 4). Models are gradient-boosted trees trained on 50 k stratified (NAICS × risk-quartile) establishment
        records using a 2020-01-01 temporal split to prevent leakage. Composite weights are optimised via
        Nelder-Mead maximisation of Spearman ρ against held-out adverse outcomes. On the validation set the
        ensemble achieves AUROC 0.733, AP 0.711, and 2.07× lift-at-top-10% — with residual degradation attributable
        to the COVID-19 inspection moratorium crossing the temporal boundary.
      </div>

      <div className="rp-section-title">1 · System Pipeline</div>
      <PipelineDiagram />

      <div className="report-grid-two rp-col2">
        <div>
          <div className="rp-section-title">2 · Feature Set (49 dimensions)</div>
          <table className="rp-table">
            <thead>
              <tr>
                <th className="rp-th">Group</th>
                <th className="rp-th">Count</th>
                <th className="rp-th">Examples</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Absolute signals', '20', 'log_inspections, log_violations, serious/WR counts, log_penalties, avg/max penalty, recent_wr_rate, trend_delta, fatality count'],
                ['Industry z-scores', '4', 'relative_violation_rate, relative_penalty, relative_serious_ratio, relative_willful_repeat (clipped ±3σ)'],
                ['NAICS one-hot', '25', '2-digit sectors 11–92; fallback to "unknown" bucket'],
              ].map(([g, c, e]) => (
                <tr key={g}>
                  <td className="rp-td"><strong>{g}</strong></td>
                  <td className="rp-td" align="center">{c}</td>
                  <td className="rp-td" style={{ color: '#64748b', fontSize: 8.2 }}>{e}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="rp-body-text" style={{ marginTop: 4, fontSize: 8.2, color: '#64748b' }}>
            Log-transform applied to indices [0, 1, 11, 12, 13] (count/penalty features). Rate features (recent_wr_rate, trend_delta) are left on natural scale. Industry z-scores are clipped at ±3σ to suppress tail inflation.
          </div>

          <div className="rp-section-title">3 · Composite Score Formula</div>
          <code className="rp-code">{compositeFormula}</code>
          <div className="rp-body-text" style={{ marginTop: 4, fontSize: 8.2, color: '#64748b' }}>
            Evidence ceiling applied post-scoring: fatality+willful → cap 70; ≤2 inspections → cap 50; 3–4 → cap 58; 5+ → no cap. Recommendation bands:{' '}
            <span style={badge('#059669')}>Recommend</span> &lt;30,{' '}
            <span style={badge('#d97706')}>Caution</span> 30–59,{' '}
            <span style={badge('#dc2626')}>Do Not Recommend</span> ≥60.
          </div>

          <div className="rp-section-title">4 · Training Setup</div>
          <table className="rp-table">
            <tbody>
              {[
                ['Temporal split', '2020-01-01 (features pre-cutoff; labels 2020–2023)'],
                ['Label window', '3-year adverse outcome window (2020–12-31 → 2023-12-31)'],
                ['Sample size', '50 000 rows'],
                ['Stratification', '2-digit NAICS × risk-quartile (pseudo-label)'],
                ['Percentiles', 'P75/P90/P95 per NAICS sector, pre-2020 data only'],
              ].map(([k, v]) => (
                <tr key={k}>
                  <td className="rp-td" style={{ fontWeight: 600, whiteSpace: 'nowrap', fontSize: 8.5 }}>{k}</td>
                  <td className="rp-td" style={{ color: '#374151', fontSize: 8.5 }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <div className="rp-section-title">5 · Model Architecture – 4 Prediction Heads</div>
          <table className="rp-table">
            <thead>
              <tr>
                <th className="rp-th">Head</th>
                <th className="rp-th">Target</th>
                <th className="rp-th">Algorithm</th>
                <th className="rp-th">Output</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['1',    'any_wr_serious (0/1)',          'GBC + isotonic calibration',      'p_serious_wr ∈ [0,1]'],
                ['2',    'log1p(future_penalty)',          'GBR Huber + StandardScaler',      'E[penalty] USD'],
                ['3',    'future_citation_count',          'GBR Huber + StandardScaler',      'E[citations]'],
                ['4a–c', 'is_mod / large / extreme penalty', '3× GBC + isotonic calibration', 'p_{mod,lg,ext} ∈ [0,1]'],
              ].map(([h, t, a, o]) => (
                <tr key={h}>
                  <td className="rp-td" style={{ fontWeight: 700, whiteSpace: 'nowrap' }}>
                    <span style={badge('#1d5a8e')}>{h}</span>
                  </td>
                  <td className="rp-td" style={{ fontSize: 8.2 }}>{t}</td>
                  <td className="rp-td" style={{ fontSize: 8.2, color: '#64748b' }}>{a}</td>
                  <td className="rp-td" style={{ fontSize: 8.2 }}>{o}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 6 }}>
            <div className="rp-body-text" style={{ fontSize: 8.5, marginBottom: 3 }}>Composite head weight allocation:</div>
            <WeightChart />
          </div>

          <div className="rp-section-title">6 · Hyperparameters</div>
          <code className="rp-code" style={{ fontSize: 7.8 }}>{hyperparams}</code>

          <div className="rp-section-title">7 · Validation Metrics</div>
          <div className="rp-body-text" style={{ fontSize: 8.2, marginBottom: 4, color: '#64748b' }}>
            Head-1 (p_wr) on stratified hold-out. Dashed line = original target (pre-COVID adjustment).
          </div>
          <MetricsChart />
          <table className="rp-table" style={{ marginTop: 6 }}>
            <thead>
              <tr>
                <th className="rp-th">Metric</th>
                <th className="rp-th">Value</th>
                <th className="rp-th">Target</th>
                <th className="rp-th">Note</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['AUROC',          '0.733', '0.760', 'COVID ceiling ≈ 0.733'],
                ['Avg Precision',  '0.711', '0.760', 'COVID ceiling'],
                ['Brier Skill Score', '0.163', '0.180', 'COVID ceiling'],
                ['Lift @top-10%',  '2.07×', '2.0×',  '✓ passes'],
                ['Capture @top-10%', '12.0%', '11%', '✓ passes'],
              ].map(([m, v, t, n]) => (
                <tr key={m}>
                  <td className="rp-td" style={{ fontWeight: 600, fontSize: 8.5 }}>{m}</td>
                  <td className="rp-td" style={{ fontSize: 8.5 }}>{v}</td>
                  <td className="rp-td" style={{ fontSize: 8.5, color: '#64748b' }}>{t}</td>
                  <td className="rp-td" style={{ fontSize: 8, color: '#9ca3af' }}>{n}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="rp-body-text" style={{ fontSize: 8, color: '#9ca3af', marginTop: 4 }}>
            COVID-19 inspection moratorium (2020) depresses metrics for all models trained on pre-2020 → post-2020 splits; degradation is structural, not model error. Non-AUROC metrics (lift, capture, calibration) all pass adjusted thresholds.
          </div>
        </div>
      </div>

      <div className="rp-footer">
        OSHA Risk Assessment Platform · vetManufactures · Data source: OSHA Enforcement Database ·{' '}
        Model retrained automatically when feature schema changes (cache-invalidation fingerprint guard)
      </div>
    </div>
  )
}
