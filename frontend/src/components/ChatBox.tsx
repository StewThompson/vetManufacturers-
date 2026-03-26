import { useRef, useState } from 'react'
import type { AssessmentResponse } from '../types/assessment'
import { askQuestion } from '../api/client'

interface Message {
  role: 'user' | 'assistant'
  text: string
}

export default function ChatBox({ result }: { result: AssessmentResponse }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', text: q }])
    setLoading(true)
    try {
      const answer = await askQuestion(q, result)
      setMessages((prev) => [...prev, { role: 'assistant', text: answer }])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `Error: ${String(err)}` },
      ])
    } finally {
      setLoading(false)
      // re-focus input after response
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') send()
  }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div className="card-title">Ask a Question</div>

      {messages.length === 0 && (
        <p style={{ fontSize: 13, color: 'var(--text-muted)', margin: 0 }}>
          Ask anything about this assessment — specific violations, OSHA standards,
          accidents, industry comparisons, or what a risk driver means.
        </p>
      )}

      {messages.length > 0 && (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            maxHeight: 420,
            overflowY: 'auto',
            paddingRight: 4,
          }}
        >
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                flexDirection: m.role === 'user' ? 'row-reverse' : 'row',
                gap: 8,
                alignItems: 'flex-start',
              }}
            >
              <div
                style={{
                  background: m.role === 'user' ? 'var(--accent)' : 'var(--surface-2)',
                  color: m.role === 'user' ? '#fff' : 'var(--text)',
                  border: m.role === 'assistant' ? '1px solid var(--border)' : 'none',
                  borderRadius: 10,
                  padding: '8px 12px',
                  fontSize: 13,
                  lineHeight: 1.6,
                  maxWidth: '85%',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {m.text}
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div className="spinner" />
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Thinking…</span>
            </div>
          )}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8 }}>
        <input
          ref={inputRef}
          className="search-input"
          placeholder="e.g. What is 1910.1200? Were there any fatalities?"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
          style={{ flex: 1 }}
        />
        <button
          className="btn btn-primary"
          onClick={send}
          disabled={loading || !input.trim()}
        >
          Ask
        </button>
      </div>
    </div>
  )
}
