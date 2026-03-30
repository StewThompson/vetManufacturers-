import type { AssessmentResponse, AssessParams, SearchResponse, SSEEvent } from '../types/assessment'

const BASE = '/api'

// ── REST helpers ─────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText)
    throw new Error(`API ${path} → ${res.status}: ${body}`)
  }
  return res.json() as Promise<T>
}

export const getCompanies = (): Promise<string[]> =>
  apiFetch('/companies')

export const searchCompanies = (q: string): Promise<SearchResponse> =>
  apiFetch(`/search?q=${encodeURIComponent(q)}`)

export const getLocations = (company: string): Promise<string[]> =>
  apiFetch(`/locations?company=${encodeURIComponent(company)}`)

// ── SSE assess ───────────────────────────────────────────────────────────────

/**
 * Opens a server-sent-event stream for an assessment.
 * Returns a cleanup function (call on unmount).
 *
 * @param params  - assessment parameters
 * @param onEvent - called for each parsed SSE event
 */
export function openAssessStream(
  params: AssessParams,
  onEvent: (evt: SSEEvent) => void,
): () => void {
  const qs = new URLSearchParams()
  if (params.company) qs.set('company', params.company)
  if (params.raw_names?.length) qs.set('raw_names', params.raw_names.join(','))
  if (params.display_name) qs.set('display_name', params.display_name)
  if (params.years_back != null) qs.set('years_back', String(params.years_back))

  const url = `${BASE}/assess?${qs.toString()}`

  // Use fetch + ReadableStream so the proxy/CORS headers are respected
  let cancelled = false
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null

  const run = async () => {
    const resp = await fetch(url, { headers: { Accept: 'text/event-stream' } })
    if (!resp.body) throw new Error('No response body')

    reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done || cancelled) break

      buffer += decoder.decode(value, { stream: true })

      // Parse complete SSE frames (delimited by double newline)
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''   // last element may be incomplete

      for (const frame of frames) {
        const eventMatch = frame.match(/^event:\s*(.+)$/m)
        const dataMatch = frame.match(/^data:\s*(.+)$/ms)
        if (!dataMatch) continue
        try {
          const payload = JSON.parse(dataMatch[1].trim())
          const eventType = eventMatch?.[1]?.trim()
          if (eventType === 'progress' || eventType === 'result' || eventType === 'error') {
            onEvent({ type: eventType, ...payload } as SSEEvent)
          }
        } catch {
          // malformed JSON — skip
        }
      }
    }
  }

  run().catch((err) => {
    if (!cancelled) onEvent({ type: 'error', message: String(err) })
  })

  return () => {
    cancelled = true
    reader?.cancel().catch(() => {})
  }
}

/** Ask a question about a completed assessment. */
export async function askQuestion(
  question: string,
  assessment: AssessmentResponse,
): Promise<string> {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, assessment }),
  })
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText)
    throw new Error(`Chat API → ${res.status}: ${body}`)
  }
  const data = await res.json() as { answer: string }
  return data.answer
}

/** Drop sites scoring above `threshold` and re-run the scoring pipeline. */
export async function recalculateDroppingHighRisk(
  assessment: AssessmentResponse,
  threshold: 30 | 60,
): Promise<AssessmentResponse> {
  const res = await fetch(`${BASE}/recalculate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ assessment, threshold }),
  })
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText)
    throw new Error(`Recalculate → ${res.status}: ${body}`)
  }
  return res.json() as Promise<AssessmentResponse>
}
