# Development startup guide

## Start the full stack

Run from `c:\Users\Stewart\vetManufactures\`:

```
scripts\start_dev.bat
```

This starts:
- **FastAPI** on `http://localhost:8000` (with `--reload`)  
- **Vite dev server** on `http://localhost:5173` (proxies `/api` to the backend)

Or start them separately in two terminals:

**Terminal 1 — API:**
```
C:\Users\Stewart\AppData\Local\Python\bin\python.exe -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```
cd frontend
npm run dev
```

## API docs (Swagger UI)
`http://localhost:8000/docs`

## Production build (frontend only)
```
cd frontend
npm run build
```
Output goes to `frontend/dist/` — serve with any static file host.

## Run existing Streamlit app (unchanged)
```
C:\Users\Stewart\AppData\Local\Python\bin\python.exe -m streamlit run app.py
```
