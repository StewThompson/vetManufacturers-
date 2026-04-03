import type { AssessmentResponse } from '../types/assessment'

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
        <div
          className="warning-box"
          style={{ background: 'var(--danger-bg)', borderColor: 'var(--danger-border)', color: 'var(--danger)' }}
        >
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
