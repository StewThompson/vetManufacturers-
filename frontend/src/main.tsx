import React from 'react'
import ReactDOM from 'react-dom/client'
import { ClientSideRowModelModule, CommunityFeaturesModule, ModuleRegistry } from 'ag-grid-community'
import App from './App'
import './index.css'
import 'ag-grid-community/styles/ag-grid.css'
import 'ag-grid-community/styles/ag-theme-alpine.css'

// AG Grid v32 requires explicit module registration
ModuleRegistry.registerModules([ClientSideRowModelModule, CommunityFeaturesModule])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
