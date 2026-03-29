import { useCallback, useRef, useState } from 'react'
import { openAssessStream } from './api/client'
import type { AssessmentResponse, SSEEvent } from './types/assessment'
import SearchCard from './components/SearchCard'
import ProgressStream, {
  ExplanationPanel,
  OutlookPanel,
  RiskBanner,
  RiskTargetsPanel,
  StatsGrid,
} from './components/AssessmentResult'
import ViolationsGrid from './components/ViolationsGrid'
import SiteBreakdown from './components/SiteBreakdown'
import ChatBox from './components/ChatBox'

type Tab = 'overview' | 'violations' | 'sites' | 'details'

export default function App() {
  const [selectedRawNames, setSelectedRawNames] = useState<string[]>([])
  const [displayName, setDisplayName] = useState<string>('')
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState<string[]>([])
  const [result, setResult] = useState<AssessmentResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('overview')
  const cleanupRef = useRef<(() => void) | null>(null)

  const handleSelectionChange = useCallback(
    (rawNames: string[], name: string) => {
      setDisplayName(name)
      setSelectedRawNames(rawNames)
    },
    [],
  )

  const handleRun = useCallback(() => {
    if (selectedRawNames.length === 0) return

    // Cancel any previous in-flight request
    cleanupRef.current?.()

    setIsRunning(true)
    setProgress([])
    setResult(null)
    setError(null)
    setTab('overview')

    const handleEvent = (evt: SSEEvent) => {
      if (evt.type === 'progress') {
        setProgress((prev) => [...prev, evt.message])
      } else if (evt.type === 'result') {
        setResult(evt.data)
        setIsRunning(false)
      } else if (evt.type === 'error') {
        setError(evt.message)
        setIsRunning(false)
      }
    }

    cleanupRef.current = openAssessStream(
      { raw_names: selectedRawNames, display_name: displayName, years_back: 10 },
      handleEvent,
    )
  }, [selectedRawNames, displayName])

  const tabs: { id: Tab; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'violations', label: `Violations${result ? ` (${result.records.flatMap((r) => r.violations).length})` : ''}` },
    { id: 'sites', label: `Sites${result ? ` (${result.establishment_count})` : ''}` },
    { id: 'details', label: 'Model Details' },
  ]

  return (
    <div className="app-shell">
      {/* Top bar */}
      <header className="topbar">
        <div className="topbar-logo">
          Manufacturer <span>Compliance</span> Intelligence
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-muted)' }}>
          OSHA Risk Assessment Platform
        </div>
      </header>

      <div className="main">
        {/* Left sidebar — search */}
        <aside className="sidebar">
          <SearchCard
            onSelectionChange={handleSelectionChange}
            isRunning={isRunning}
            onRun={handleRun}
          />
        </aside>

        {/* Right content — results */}
        <main className="content">
          {/* Progress stream */}
          <ProgressStream
            messages={progress}
            isRunning={isRunning}
            result={result}
            error={error}
          />

          {!result && !isRunning && progress.length === 0 && (
            <div className="empty-state">
              <div className="empty-state-icon">🏭</div>
              <div className="empty-state-title">No assessment loaded</div>
              <div className="empty-state-desc">
                Search for a manufacturer on the left, select facilities, and click
                "Run Assessment" to generate an OSHA risk profile.
              </div>
            </div>
          )}

          {result && (
            <>
              <RiskBanner result={result} />
              <StatsGrid result={result} />

              <div className="tabs">
                {tabs.map((t) => (
                  <button
                    key={t.id}
                    className={`tab-btn ${tab === t.id ? 'active' : ''}`}
                    onClick={() => setTab(t.id)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {tab === 'overview' && (
                <>
                  {result.risk_targets && (
                    <RiskTargetsPanel targets={result.risk_targets} />
                  )}
                  {result.outlook && (
                    <OutlookPanel outlook={result.outlook} />
                  )}
                  <div className="card">
                    <div className="card-title">Summary</div>
                    <p className="explanation-text">{result.explanation}</p>
                  </div>
                </>
              )}

              {tab === 'violations' && (
                <div className="card" style={{ padding: 0 }}>
                  <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
                    <div className="card-title" style={{ marginBottom: 0 }}>
                      All Violations — {result.records.flatMap((r) => r.violations).length} records
                    </div>
                  </div>
                  <div style={{ padding: 16 }}>
                    <ViolationsGrid records={result.records} />
                  </div>
                </div>
              )}

              {tab === 'sites' && (
                <div className="card" style={{ padding: 0 }}>
                  <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
                    <div className="card-title" style={{ marginBottom: 0 }}>
                      Site-Level Risk Scores
                    </div>
                  </div>
                  <div style={{ padding: 16 }}>
                    <SiteBreakdown sites={result.site_scores} />
                  </div>
                </div>
              )}

              {tab === 'details' && <ExplanationPanel result={result} />}

              <ChatBox result={result} />
            </>
          )}
        </main>
      </div>
    </div>
  )
}
