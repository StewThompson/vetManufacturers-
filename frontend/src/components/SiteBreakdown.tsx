import { useMemo } from 'react'
import { AgGridReact } from 'ag-grid-react'
import type { ColDef } from 'ag-grid-community'
import type { SiteScoreOut } from '../types/assessment'

function scoreStyle(params: { value: number }): Record<string, string | number> {
  const v = params.value as number
  if (v >= 0.7) return { color: '#dc2626', fontWeight: 700 }
  if (v >= 0.4) return { color: '#d97706', fontWeight: 500 }
  return { color: '#059669', fontWeight: 400 }
}

const COLUMN_DEFS: ColDef<SiteScoreOut>[] = [
  { field: 'name', headerName: 'Site', flex: 1, filter: 'agTextColumnFilter' },
  { field: 'city', headerName: 'City', width: 140, filter: 'agTextColumnFilter' },
  { field: 'state', headerName: 'State', width: 80, filter: 'agTextColumnFilter' },
  { field: 'naics_code', headerName: 'NAICS', width: 100 },
  { field: 'n_inspections', headerName: 'Inspections', width: 120, type: 'numericColumn', filter: 'agNumberColumnFilter' },
  {
    field: 'score',
    headerName: 'Risk Score',
    width: 120,
    type: 'numericColumn',
    filter: 'agNumberColumnFilter',
    cellStyle: scoreStyle,
    valueFormatter: (p) => (p.value as number).toFixed(3),
    sort: 'desc',
  },
]

export default function SiteBreakdown({ sites }: { sites: SiteScoreOut[] }) {
  const sorted = useMemo(
    () => [...sites].sort((a, b) => b.score - a.score),
    [sites],
  )

  return (
    <div className="ag-theme-alpine" style={{ height: 340, width: '100%' }}>
      <AgGridReact<SiteScoreOut>
        rowData={sorted}
        columnDefs={COLUMN_DEFS}
        defaultColDef={{ resizable: true, sortable: true }}
        rowHeight={32}
        headerHeight={36}
        suppressCellFocus
      />
    </div>
  )
}
