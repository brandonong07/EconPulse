# EconPulse

2026 HackDavis

EconPulse is a student-centered economic dashboard that turns public FRED
economic indicators into practical financial awareness for students and young
adults.

## Tech Stack

- Frontend: React 18, TypeScript, Vite, Recharts, lucide-react, CSS
- Backend: Python, pandas, NumPy, scikit-learn, optional XGBoost
- Data: FRED API when available, cached/mock data fallback
- Outputs: JSON files consumed by the frontend from `frontend/public/data`

## Run Backend

```powershell
py -m backend.run_pipeline
```

The backend writes processed files to `data/processed` and syncs them into
`frontend/public/data`.

## Run Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Dashboard Tabs

- Dashboard: latest economic health and category scores
- Model Hub: Linear Regression, Ridge Regression, and XGBoost forecast comparison
- What-If: slider-based scenario simulator
- Documentation: project stack, pipeline, data files, and model notes

## Model Notes

The model hub forecasts overall economic health three months ahead. Simple
shortcut forecasts were removed so the app focuses on trained models and their
validation metrics. In the simulator, switching models changes only the
predicted overall health card; the category inputs remain fixed so model
differences are easy to compare.
