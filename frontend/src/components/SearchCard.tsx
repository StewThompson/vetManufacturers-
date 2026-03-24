import { useEffect, useMemo, useRef, useState } from 'react'
import { searchCompanies } from '../api/client'
import type { FacilityOut, GroupedCompanyOut } from '../types/assessment'

// NAICS 2-digit sector labels (merged where prefix maps to same name)
const NAICS_LABELS: Record<string, string> = {
  '11': 'Agriculture', '21': 'Mining', '22': 'Utilities', '23': 'Construction',
  '31': 'Manufacturing', '32': 'Manufacturing', '33': 'Manufacturing',
  '42': 'Wholesale Trade', '44': 'Retail Trade', '45': 'Retail Trade',
  '48': 'Transportation', '49': 'Transportation',
  '51': 'Information', '52': 'Finance', '53': 'Real Estate',
  '54': 'Professional Services', '55': 'Management', '56': 'Admin Services',
  '61': 'Education', '62': 'Health Care', '71': 'Arts & Entertainment',
  '72': 'Food Services', '81': 'Other Services', '92': 'Public Administration',
}

function naicsLabel(code: string): string {
  return NAICS_LABELS[code.slice(0, 2)] ?? `NAICS ${code.slice(0, 2)}`
}

interface Props {
  onSelectionChange: (rawNames: string[], displayName: string) => void
  yearsBack: number
  onYearsBackChange: (v: number) => void
  isRunning: boolean
  onRun: () => void
}

export default function SearchCard({
  onSelectionChange,
  yearsBack,
  onYearsBackChange,
  isRunning,
  onRun,
}: Props) {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<{
    top: GroupedCompanyOut | null
    others: GroupedCompanyOut[]
  } | null>(null)
  const [loading, setLoading] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [selectedGroups, setSelectedGroups] = useState<GroupedCompanyOut[]>([])
  const [excludedNaics, setExcludedNaics] = useState<Set<string>>(new Set())
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const searchCardRef = useRef<HTMLDivElement>(null)
  const [panelTop, setPanelTop] = useState(0)

  // Compute floating panel Y position whenever results change
  useEffect(() => {
    if (!searchResults || !searchCardRef.current) return
    const rect = searchCardRef.current.getBoundingClientRect()
    setPanelTop(rect.top)
  }, [searchResults])

  // All facilities from all selected groups
  const allFacilities = useMemo<FacilityOut[]>(() => {
    const out: FacilityOut[] = []
    for (const g of selectedGroups) {
      out.push(...g.high_confidence, ...g.medium_confidence, ...g.low_confidence)
    }
    return out
  }, [selectedGroups])

  // Unique NAICS sectors present in selected facilities
  const naicsSectors = useMemo(() => {
    const seen = new Map<string, number>() // label -> count
    for (const f of allFacilities) {
      if (f.naics_code?.length >= 2) {
        const label = naicsLabel(f.naics_code)
        seen.set(label, (seen.get(label) ?? 0) + 1)
      }
    }
    return Array.from(seen.entries()).map(([label, count]) => ({ label, count }))
  }, [allFacilities])

  // Facilities after exclusion filter
  const filteredRawNames = useMemo<string[]>(() => {
    return allFacilities
      .filter((f) => {
        if (!f.naics_code || excludedNaics.size === 0) return true
        return !excludedNaics.has(naicsLabel(f.naics_code))
      })
      .map((f) => f.raw_name)
  }, [allFacilities, excludedNaics])

  // Notify parent whenever filtered selection changes
  useEffect(() => {
    const name = selectedGroups.map((g) => g.parent_name).join(' + ')
    onSelectionChange(filteredRawNames, name)
  }, [filteredRawNames]) // eslint-disable-line react-hooks/exhaustive-deps

  // Debounced search
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (query.trim().length < 2) {
      setSearchResults(null)
      return
    }
    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      setSearchError(null)
      try {
        const resp = await searchCompanies(query)
        setSearchResults({ top: resp.top_group, others: resp.other_groups })
      } catch (e) {
        setSearchError(String(e))
      } finally {
        setLoading(false)
      }
    }, 350)
  }, [query])

  const isGroupSelected = (g: GroupedCompanyOut) =>
    selectedGroups.some((s) => s.parent_name === g.parent_name)

  const toggleGroup = (group: GroupedCompanyOut) => {
    setSelectedGroups((prev) => {
      const exists = prev.some((g) => g.parent_name === group.parent_name)
      return exists
        ? prev.filter((g) => g.parent_name !== group.parent_name)
        : [...prev, group]
    })
  }

  const renderGroupRow = (group: GroupedCompanyOut, isTop: boolean) => {
    const selected = isGroupSelected(group)
    return (
      <div
        key={group.parent_name}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '10px 12px',
          borderRadius: 8,
          border: selected ? '1px solid var(--accent)' : '1px solid var(--border)',
          background: selected ? 'var(--accent-dim)' : isTop ? 'var(--surface-2)' : 'transparent',
          marginBottom: 5,
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: isTop ? 600 : 500, fontSize: isTop ? 14 : 13 }}>
            🏢 {group.parent_name}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            {group.total_facilities} establishment{group.total_facilities !== 1 ? 's' : ''}
            {' · '}
            <span className={`conf-${group.confidence_label.toLowerCase()}`}>
              {group.confidence_label} confidence
            </span>
          </div>
        </div>
        <button
          className={`btn btn-sm ${selected ? 'btn-primary' : 'btn-ghost'}`}
          onClick={() => toggleGroup(group)}
          style={{ flexShrink: 0 }}
        >
          {selected ? '✓ Added' : (isTop && selectedGroups.length === 0 ? 'Select →' : '+ Add')}
        </button>
      </div>
    )
  }

  return (
    <>
      {/* Search input */}
      <div className="card" ref={searchCardRef}>
        <div className="card-title">Manufacturer Lookup</div>
        <input
          className="search-input"
          style={{ width: '100%' }}
          placeholder="Type company name…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isRunning}
        />
        {loading && (
          <div style={{ marginTop: 8, color: 'var(--text-muted)', fontSize: 12 }}>Searching…</div>
        )}
        {searchError && (
          <div style={{ marginTop: 8, color: 'var(--danger)', fontSize: 12 }}>{searchError}</div>
        )}
      </div>

      {/* Selected groups summary */}
      {selectedGroups.length > 0 && (
        <div
          style={{
            background: '#ecfdf5',
            border: '1px solid #a7f3d0',
            borderRadius: 8,
            padding: '10px 14px',
          }}
        >
          <div style={{ color: 'var(--success)', fontWeight: 600, fontSize: 13, marginBottom: 8 }}>
            ✓ {selectedGroups.length} group{selectedGroups.length > 1 ? 's' : ''} selected:{' '}
            {selectedGroups.map((g) => g.parent_name).join(' + ')}
          </div>
          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
            {selectedGroups.map((g) => (
              <span
                key={g.parent_name}
                style={{
                  background: 'var(--surface-2)',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  padding: '2px 8px',
                  fontSize: 12,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 5,
                }}
              >
                {g.parent_name}
                <button
                  style={{
                    background: 'none', border: 'none', color: 'var(--text-muted)',
                    cursor: 'pointer', padding: 0, lineHeight: 1, fontSize: 14,
                  }}
                  onClick={() => toggleGroup(g)}
                  title={`Remove ${g.parent_name}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
          {filteredRawNames.length !== allFacilities.length && (
            <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
              {filteredRawNames.length} of {allFacilities.length} facilities after industry filter
            </div>
          )}
        </div>
      )}

      {/* Search results — floats outside the sidebar as a fixed panel */}
      {searchResults && (
        <div
          className="search-results-panel"
          style={{
            left: 368,
            top: panelTop,
            width: 420,
            maxHeight: `calc(100vh - ${panelTop}px - 16px)`,
          }}
        >
          <div style={{
            padding: '10px 14px 8px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <div className="card-title" style={{ marginBottom: 0 }}>Search Results</div>
            <button
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: 'var(--text-muted)', fontSize: 18, lineHeight: 1, padding: '0 2px',
              }}
              onClick={() => setSearchResults(null)}
              title="Close"
            >×</button>
          </div>
          <div style={{ padding: '10px 12px', overflowY: 'auto', flex: 1 }}>
            {searchResults.top && (
              <>
                <div style={{
                  fontSize: 10, fontWeight: 700, letterSpacing: 1,
                  textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 6,
                }}>
                  Best Match
                </div>
                {renderGroupRow(searchResults.top, true)}
              </>
            )}
            {searchResults.others.length > 0 && (
              <>
                <div style={{
                  fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase',
                  color: 'var(--text-muted)', margin: '12px 0 6px',
                }}>
                  Other Possible Matches
                </div>
                {searchResults.others.map((g) => renderGroupRow(g, false))}
              </>
            )}
            {!searchResults.top && searchResults.others.length === 0 && (
              <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '8px 0' }}>
                No matches found.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Industry filter — shown when 2+ NAICS sectors are present */}
      {naicsSectors.length > 1 && (
        <div className="card">
          <div className="card-title">Filter Industries</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
            Uncheck to exclude facilities from that sector.
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {naicsSectors.map(({ label, count }) => (
              <label
                key={label}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  cursor: 'pointer', fontSize: 13,
                  opacity: excludedNaics.has(label) ? 0.45 : 1,
                }}
              >
                <input
                  type="checkbox"
                  checked={!excludedNaics.has(label)}
                  onChange={() =>
                    setExcludedNaics((prev) => {
                      const next = new Set(prev)
                      next.has(label) ? next.delete(label) : next.add(label)
                      return next
                    })
                  }
                  style={{ accentColor: 'var(--accent)', width: 14, height: 14 }}
                />
                <span style={{ flex: 1 }}>{label}</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{count}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Years back + run button */}
      <div className="card">
        <div className="card-title">Assessment Window</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <label style={{ fontSize: 12, color: 'var(--text-muted)', flex: 1 }}>Years back</label>
          <select
            className="select-input"
            style={{ width: 'auto', minWidth: 80 }}
            value={yearsBack}
            onChange={(e) => onYearsBackChange(Number(e.target.value))}
            disabled={isRunning}
          >
            {[3, 5, 7, 10, 15].map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>
        <button
          className="btn btn-primary btn-full"
          disabled={isRunning || filteredRawNames.length === 0}
          onClick={onRun}
        >
          {isRunning ? 'Running assessment…' : 'Run Assessment →'}
        </button>
        {filteredRawNames.length > 0 && (
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
            {selectedGroups.length} group{selectedGroups.length !== 1 ? 's' : ''}
            {' · '}
            ~{filteredRawNames.length} establishments
          </div>
        )}
      </div>
    </>
  )
}
