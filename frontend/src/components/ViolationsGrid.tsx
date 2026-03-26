import { useMemo } from 'react'
import { AgGridReact } from 'ag-grid-react'
import type { ColDef } from 'ag-grid-community'
import type { ViolationOut, OSHARecordOut } from '../types/assessment'

interface FlatViolation extends ViolationOut {
  inspection_id: string
  date_opened: string
  site_name: string
  site_state: string
}

function flattenViolations(records: OSHARecordOut[]): FlatViolation[] {
  const rows: FlatViolation[] = []
  for (const r of records) {
    for (const v of r.violations) {
      rows.push({
        ...v,
        inspection_id: r.inspection_id,
        date_opened: r.date_opened,
        site_name: r.estab_name ?? '',
        site_state: r.site_state ?? '',
      })
    }
  }
  return rows
}

function currencyFormatter(params: { value: number }): string {
  if (params.value == null) return ''
  return `$${params.value.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
}

function boolRenderer(params: { value: boolean }) {
  return params.value ? '✓' : ''
}

function severityStyle(params: { value: string }): Record<string, string | number> {
  const severity = params.value?.toLowerCase()
  if (severity === 'willful') return { color: '#dc2626', fontWeight: 700 }
  if (severity === 'serious') return { color: '#d97706', fontWeight: 500 }
  if (severity === 'repeat') return { color: '#ea580c', fontWeight: 500 }
  return { fontWeight: 400 }
}

const COLUMN_DEFS: ColDef<FlatViolation>[] = [
  {
    field: 'date_opened',
    headerName: 'Date',
    width: 100,
    sort: 'desc',
    filter: 'agDateColumnFilter',
  },
  {
    field: 'site_name',
    headerName: 'Site',
    flex: 2,
    minWidth: 140,
    filter: 'agTextColumnFilter',
  },
  {
    field: 'site_state',
    headerName: 'St.',
    width: 54,
    filter: 'agTextColumnFilter',
  },
  {
    field: 'category',
    headerName: 'Standard',
    width: 120,
    filter: 'agTextColumnFilter',
  },
  {
    field: 'severity',
    headerName: 'Severity',
    width: 100,
    filter: 'agTextColumnFilter',
    cellStyle: severityStyle,
  },
  {
    field: 'penalty_amount',
    headerName: 'Penalty',
    width: 100,
    filter: 'agNumberColumnFilter',
    valueFormatter: currencyFormatter,
    type: 'numericColumn',
  },
  {
    field: 'is_willful',
    headerName: 'Willful',
    width: 72,
    valueFormatter: boolRenderer as never,
    type: 'numericColumn',
  },
  {
    field: 'is_repeat',
    headerName: 'Repeat',
    width: 72,
    valueFormatter: boolRenderer as never,
    type: 'numericColumn',
  },
  {
    field: 'gravity',
    headerName: 'Grav.',
    width: 66,
    filter: 'agTextColumnFilter',
  },
  {
    field: 'nr_exposed',
    headerName: 'Exposed',
    width: 82,
    filter: 'agNumberColumnFilter',
    type: 'numericColumn',
  },
  {
    field: 'description',
    headerName: 'Description',
    flex: 1,
    minWidth: 120,
    filter: 'agTextColumnFilter',
    tooltipField: 'description',
    wrapText: false,
  },
]

interface Props {
  records: OSHARecordOut[]
}

export default function ViolationsGrid({ records }: Props) {
  const rowData = useMemo(() => flattenViolations(records), [records])

  return (
    <div
      className="ag-theme-alpine"
      style={{ height: 520, width: '100%' }}
    >
      <AgGridReact<FlatViolation>
        rowData={rowData}
        columnDefs={COLUMN_DEFS}
        defaultColDef={{
          resizable: true,
          sortable: true,
          filter: true,
        }}
        pagination
        paginationPageSize={50}
        paginationPageSizeSelector={[25, 50, 100, 200]}
        rowHeight={32}
        headerHeight={36}
        suppressCellFocus
        tooltipShowDelay={300}
        enableCellTextSelection
      />
    </div>
  )
}
