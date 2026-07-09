# EconPulse

**Live Demo:** [econ-pulse-9cuc.vercel.app](https://econ-pulse-9cuc.vercel.app/)

A student-centered economic dashboard that converts public FRED macroeconomic data into practical financial awareness tools for students, young adults, and early investors.

Built for HackDavis 2026, EconPulse combines data pipelines, machine learning models, scenario simulation, and an interactive frontend to make economic conditions easier to interpret.

## Project overview

Economic indicators are often scattered across government dashboards, dense reports, and disconnected datasets. EconPulse brings these indicators into one interface and translates them into a more intuitive view of economic health.

The project focuses on questions like:

- Is the labor market strengthening or weakening?
- Is inflation pressure rising or cooling?
- How do different indicators combine into an overall economic health score?
- How do model predictions change under different economic scenarios?

## Screenshots

### Model Hub

![Model Hub](docs/screenshots/model-hub.png)

### Model Cards

![Multiple Linear Regression](docs/screenshots/multi-linear-regression.png)

![XGBoost](docs/screenshots/xgboost.png)

![LightGBM](docs/screenshots/lightgbm.png)

![MLP Neural Network](docs/screenshots/mlp-neural-network.png)

## My Contribution

- Built backend data processing scripts using Python, pandas, and NumPy
- Helped design the economic health scoring logic and model comparison workflow
- Integrated processed model outputs into frontend-ready JSON files
- Contributed to the Model Hub and scenario simulation logic
- Documented model performance and project limitations

## Key features

- Economic dashboard using FRED-style macro indicators
- Processed data pipeline with cached/mock fallback support
- Model Hub comparing Multiple Linear Regression, XGBoost, LightGBM, and an MLP Neural Network
- Scenario simulator for changing category assumptions
- Interactive frontend built with React, TypeScript, Vite, and Recharts
- Backend pipeline using Python, pandas, NumPy, and scikit-learn

## Tech stack

**Frontend**
- React
- TypeScript
- Vite
- Recharts
- lucide-react
- CSS

**Backend**
- Python
- pandas
- NumPy
- scikit-learn
- Optional XGBoost

## Data sources

- EconPulse uses public macroeconomic indicators from FRED where available, with cached/mock fallback data for development and demo stability.
- The latest metric snapshot includes values from February 2026 through May 2026, depending on each series' latest available release date.

**Data**
- FRED API when available
- Cached or mock data fallback
- Processed JSON files served to the frontend

## Indicators Used

EconPulse organizes FRED-style macroeconomic indicators into six economic pressure categories:

### Rent Pressure
- Rent CPI (`CUSR0000SEHA`)
- Shelter CPI (`CUSR0000SAH1`)
- 30-Year Mortgage Rate (`MORTGAGE30US`)
- Federal Funds Rate (`FEDFUNDS`)
- Real Wages (`AHETPI`)
- U.S. National Home Price Index (`CSUSHPISA`)

### Job Market
- Unemployment Rate (`UNRATE`)
- Job Openings (`JTSJOL`)
- Initial Jobless Claims (`ICSA`)
- Labor Force Participation Rate (`CIVPART`)
- Average Hourly Earnings (`CES0500000003`)

### Inflation Pressure
- CPI: All Items (`CPIAUCSL`)
- Core CPI (`CPILFESL`)
- Food CPI (`CPIUFDSL`)
- Energy CPI (`CPIENGSL`)
- Regular Gas Prices (`GASREGCOVW`)

### Borrowing Pressure
- Federal Funds Rate (`FEDFUNDS`)
- Credit Card Interest Rate (`TERMCBCCALLNS`)
- 30-Year Mortgage Rate (`MORTGAGE30US`)

### Wage Strength
- Average Hourly Earnings (`CES0500000003`)
- Real Hourly Earnings (`CES0500000011`)
- CPI Inflation (`CPIAUCSL`)

### Consumer Sentiment
- University of Michigan Consumer Sentiment Index (`UMCSENT`)

These indicators are processed into category-level scores and an overall economic health score. The dashboard uses the latest available values and year-over-year changes to help users interpret macroeconomic conditions across affordability, labor, inflation, borrowing, wages, and sentiment.

## Repository structure

```text
EconPulse/
├── backend/
├── data/
├── frontend/
│   └── public/data/
└── README.md
```

## Run the backend

```bash
python -m backend.run_pipeline
```

The backend writes processed files to `data/processed` and syncs frontend-ready outputs into:

```text
frontend/public/data
```

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

## Dashboard sections

| Section | Purpose |
|---|---|
| Dashboard | Displays current economic health and category-level scores |
| Model Hub | Compares model forecasts and validation metrics |
| What-If Simulator | Tests scenario changes with slider-based inputs |
| Documentation | Explains pipeline, stack, and data files |

## Modeling approach

The model hub forecasts overall economic health three months ahead. The project compares multiple supervised learning models so users can see how different model assumptions affect predictions.

Current model comparison includes:

- Multiple Linear Regression
- XGBoost
- LightGBM
- MLP Neural Network

## Model Performance

The Model Hub compares several supervised learning models for a 3-month economic health forecast.

| Model | Forecasted Health | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Multiple Linear Regression | 41.8 | 2.72 | 3.29 | 0.871 |
| XGBoost | 41.6 | 2.06 | 2.51 | 0.925 |
| LightGBM | 42.1 | 2.08 | 2.41 | 0.930 |
| MLP Neural Network | 39.7 | 2.38 | 2.84 | 0.903 |

Lower MAE and RMSE indicate smaller forecast errors. Higher R² indicates that the model explains more variation in the future economic health target.

LightGBM performed best overall, with the lowest RMSE and highest R² among the tested models. XGBoost had the lowest MAE, making both gradient-boosting models stronger than the linear and neural-network baselines.

## What I learned

This project helped me connect economics, statistics, and software engineering in one system. I practiced building an end-to-end data product, from data ingestion and preprocessing to modeling, frontend design, and deployment.

## Planned improvements

- Add more economic indicators and metadata
- Improve model validation and backtesting
- Add time-series models
- Add confidence intervals or uncertainty bands
- Improve responsive design
- Add richer documentation for each indicator
- Add automated tests for backend data transformations

## Limitations

- This project is educational and should not be used as financial advice.
- Model outputs depend on the selected indicators and available historical data.
- The economic health score is a simplified proxy and does not capture every macroeconomic condition.
- Forecasts are sensitive to feature engineering choices and may not generalize across all regimes.

## Candidate signal

EconPulse reflects my interest in quantitative finance, macroeconomic data, and applied machine learning. I built this project to show that I can work across data pipelines, statistical modeling, and frontend product design.
