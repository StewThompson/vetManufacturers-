import type { AssessmentResponse } from '../types/assessment'

interface Props {
  messages: string[]
  isRunning: boolean
  result: AssessmentResponse | null
  error: string | null
}

export default function ProgressStream({ messages, isRunning, result, error }: Props) {
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
