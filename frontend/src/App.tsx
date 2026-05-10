import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Briefcase,
  CreditCard,
  DollarSign,
  Gauge,
  Home,
  LineChart as LineChartIcon,
  ShoppingCart,
  SlidersHorizontal,
  Smile,
  TrendingDown,
  TrendingUp,
  Wallet,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type Tab = 'dashboard' | 'models' | 'simulator';

type ScoreRow = {
  date: string;
  overall_health: number | null;
  macro_state_label: string | null;
  rent_pressure: number | null;
  job_market_strength: number | null;
  inflation_pressure: number | null;
  borrowing_pressure: number | null;
  wage_strength: number | null;
  consumer_sentiment: number | null;
};

type KpiSparkPoint = {
  time: number;
  value: number | null;
  forecast?: boolean;
};

type ModelMetric = {
  mae: number;
  rmse: number;
  r2: number | null;
};

type ModelResult = {
  model: string;
  backend: string;
  metrics?: ModelMetric;
  prediction?: number;
  description?: string;
  feature_set?: string;
  feature_count?: number;
  status?: string;
  reason?: string;
};

type ArtifactModel = {
  key: string;
  label: string;
  status: string;
  prediction?: number;
  metrics?: ModelMetric;
  model_class?: string;
  feature_names?: string[];
  reason?: string;
  sensitivity?: Record<string, { base_value: number; approx_slope_per_point: number }>;
};

type LoadedData = {
  dashboard: any;
  scores: ScoreRow[];
  modelResults: any;
  artifacts: any;
  latest: any;
};

const dataFiles = {
  dashboard: '/data/dashboard_metrics.json',
  scores: '/data/scores.json',
  modelResults: '/data/model_results.json',
  artifacts: '/data/model_artifacts.json',
  latest: '/data/latest_metrics.json',
};

const dataVersion = Date.now().toString();

const categoryLabels: Record<string, string> = {
  rent_pressure: 'Rent Pressure',
  job_market_strength: 'Job Market',
  inflation_pressure: 'Inflation',
  borrowing_pressure: 'Borrowing',
  wage_strength: 'Wages',
  consumer_sentiment: 'Sentiment',
};

const scoreRanges = [
  { range: '0-20', label: 'Severe Stress', detail: 'crisis conditions' },
  { range: '20-40', label: 'Strained', detail: 'weak conditions' },
  { range: '40-50', label: 'Fragile', detail: 'uneven recovery' },
  { range: '50-60', label: 'Stable', detail: 'steady economy' },
  { range: '60-80', label: 'Healthy', detail: 'broad growth' },
  { range: '80-100', label: 'Strong', detail: 'strong expansion' },
];

const featureDescriptions: Record<string, string> = {
  rent_pressure: 'Higher means housing is more affordable and rent pressure is lower.',
  job_market_strength: 'Higher means stronger hiring, lower unemployment, and more labor demand.',
  inflation_pressure: 'Higher means prices are more stable. Lower means inflation is more painful.',
  borrowing_pressure: 'Higher means credit conditions are easier. Lower means rates and debt are more expensive.',
  wage_strength: 'Higher means wages are stronger and more likely to keep up with costs.',
  consumer_sentiment: 'Higher means households feel more confident. Lower means pessimism and stress.',
};

const sliderEffects: Record<string, string> = {
  rent_pressure: 'Moving this down raises student cost pressure because rent is a major student expense.',
  job_market_strength: 'Moving this up improves macro health because students face better job prospects.',
  inflation_pressure: 'This is not raw inflation. Moving it down means worse inflation conditions.',
  borrowing_pressure: 'Moving this down makes loans, credit cards, and financing feel more restrictive.',
  wage_strength: 'Moving this up supports affordability because income conditions improve.',
  consumer_sentiment: 'Moving this down signals household stress, which weighs on overall macro health.',
};

const modelDescriptions: Record<string, string> = {
  linear_regression:
    'Autoregressive linear model: uses current, lagged, and rolling pressure history to forecast near-term economic stress.',
  ridge_regression:
    'Regularized macro model: uses the current category scores with coefficient shrinkage to reduce overfitting.',
  xgboost_regressor:
    'Nonlinear tree model: tests whether interactions across lagged macro features improve the pressure forecast.',
};

const modelDisplayNames: Record<string, string> = {
  linear_regression: 'Linear Regression',
  ridge_regression: 'Ridge Regression',
  xgboost_regressor: 'XGBoost Regressor',
};

const modelShortDescriptions: Record<string, string> = {
  linear_regression: 'Fits a straight-line relationship from recent macro history.',
  ridge_regression: 'Fits a regularized straight-line macro relationship.',
  xgboost_regressor: 'Uses nonlinear trees to test richer macro interactions.',
};

const artifactDescriptions: Record<string, string> = {
  multiple_linear_regression:
    'Transparent model that connects the six category scores to forecasted economic health.',
  xgboost_saved:
    'Tree-based model that captures nonlinear relationships between economic categories.',
  lightgbm_saved:
    'Fast gradient-boosting model used as an alternate nonlinear forecast.',
  mlp_saved:
    'Neural network model that can learn curved relationships, but is less transparent.',
};

const macroWeights: Record<string, number> = {
  job_market_strength: 0.3,
  consumer_sentiment: 0.2,
  inflation_pressure: 0.18,
  wage_strength: 0.12,
  rent_pressure: 0.1,
  borrowing_pressure: 0.1,
};

const pressureWeights: Record<string, number> = {
  rent_pressure: 0.35,
  inflation_pressure: 0.3,
  borrowing_pressure: 0.2,
  wage_strength: 0.15,
};

function clamp(value: number, min = 0, max = 100) {
  return Math.min(max, Math.max(min, value));
}

function fmt(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  return value.toFixed(digits);
}

function monthTimestamp(date: string) {
  const [year, month] = date.slice(0, 7).split('-').map(Number);
  return Date.UTC(year, month - 1, 1);
}

function formatYearTick(value: number) {
  return String(new Date(value).getUTCFullYear());
}

function formatMonthLabel(value: number) {
  const date = new Date(value);
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  return `${year}-${month}`;
}

function modelName(name: string) {
  if (modelDisplayNames[name]) return modelDisplayNames[name];
  return name
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function sourceLabel(source: string) {
  if (source === 'fred') return 'FRED Live';
  if (source === 'cache') return 'FRED Data';
  if (source === 'legacy_cache') return 'FRED Data';
  if (source === 'mock') return 'Demo Data';
  return source.toUpperCase();
}

function sourceDetail(source: string, asOf: string, refreshedAt?: string) {
  const refreshText = refreshedAt ? `Refreshed ${refreshedAt}` : 'Refreshed this run';
  const scoreText = `latest complete score month ${asOf}`;
  if (source === 'fred') return `${refreshText}; ${scoreText}`;
  if (source === 'cache') return `${refreshText}; ${scoreText}`;
  if (source === 'legacy_cache') return `${refreshText}; ${scoreText}`;
  return `${refreshText}; ${scoreText}`;
}

function calculateMacroHealth(values: Record<string, number>) {
  const weightedBase = Object.entries(macroWeights).reduce(
    (total, [feature, weight]) => total + (values[feature] ?? 0) * weight,
    0,
  );

  let penalty = 0;
  const job = values.job_market_strength ?? 0;
  penalty += Math.max(0, 40 - job) * 0.65;
  penalty += Math.max(0, 30 - job) * 0.5;
  penalty += Math.max(0, 40 - (values.consumer_sentiment ?? 0)) * 0.22;
  penalty += Math.max(0, 35 - (values.inflation_pressure ?? 0)) * 0.28;
  penalty += Math.max(0, 35 - (values.borrowing_pressure ?? 0)) * 0.16;
  penalty += Math.max(0, 35 - (values.rent_pressure ?? 0)) * 0.16;

  const weakCount = Object.keys(macroWeights).filter((feature) => (values[feature] ?? 0) < 40).length;
  penalty += Math.max(0, weakCount - 2) * 2;

  const raw = weightedBase - penalty;
  return clamp(50 + (raw - 50) * 1.5);
}

function calculateStudentCostPressure(values: Record<string, number>) {
  const weighted = Object.entries(pressureWeights).reduce(
    (total, [feature, weight]) => total + (100 - (values[feature] ?? 0)) * weight,
    0,
  );
  const totalWeight = Object.values(pressureWeights).reduce((total, weight) => total + weight, 0);
  return clamp(weighted / totalWeight);
}

function scoreTone(score: number | null | undefined) {
  if (score === null || score === undefined) return 'neutral';
  if (score < 20) return 'danger';
  if (score < 40) return 'warning';
  if (score < 60) return 'neutral';
  return 'good';
}

function metricBand(score: number | null | undefined, inverse = false, override?: string | null) {
  if (override) return { label: override, tone: scoreTone(score) };
  if (score === null || score === undefined || Number.isNaN(score)) return { label: 'Unknown', tone: 'neutral' };

  const adjusted = inverse ? 100 - score : score;
  if (adjusted < 20) return { label: 'Poor', tone: 'danger' };
  if (adjusted < 40) return { label: 'Low', tone: 'warning' };
  if (adjusted < 60) return { label: 'Moderate', tone: 'neutral' };
  if (adjusted < 80) return { label: 'Good', tone: 'good' };
  return { label: 'Strong', tone: 'good' };
}

function trendTone(change: number | null | undefined, inverse = false) {
  if (change === null || change === undefined || Math.abs(change) < 0.05) return 'flat';
  const goodMove = inverse ? change < 0 : change > 0;
  return goodMove ? 'good' : 'bad';
}

function trendLabel(change: number | null | undefined) {
  if (change === null || change === undefined || Number.isNaN(change)) return '0.0';
  return `${change > 0 ? '+' : ''}${fmt(change)}`;
}

function pressureBand(score: number | null | undefined) {
  if (score === null || score === undefined || Number.isNaN(score)) return { label: 'Unknown', tone: 'neutral' };
  if (score < 25) return { label: 'Low pressure', tone: 'good' };
  if (score < 50) return { label: 'Moderate pressure', tone: 'neutral' };
  if (score < 75) return { label: 'Elevated pressure', tone: 'warning' };
  return { label: 'High pressure', tone: 'danger' };
}

function scoreHue(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return 205;
  return Math.round((clamp(value) / 100) * 125);
}

function scoreFill(value: number | null | undefined) {
  const hue = scoreHue(value);
  return `linear-gradient(90deg, hsl(${hue} 72% 58%), hsl(${Math.min(hue + 12, 132)} 70% 68%))`;
}

function latestChange(scores: ScoreRow[], key: keyof ScoreRow) {
  const usable = scores
    .filter((row) => typeof row[key] === 'number')
    .map((row) => Number(row[key]));
  if (usable.length < 2) return 0;
  return usable[usable.length - 1] - usable[usable.length - 2];
}

function scoreValue(row: ScoreRow, key: keyof ScoreRow | 'student_cost_pressure') {
  if (key === 'student_cost_pressure') {
    if (
      row.rent_pressure === null ||
      row.inflation_pressure === null ||
      row.borrowing_pressure === null ||
      row.wage_strength === null
    ) {
      return null;
    }
    return calculateStudentCostPressure({
      rent_pressure: row.rent_pressure,
      inflation_pressure: row.inflation_pressure,
      borrowing_pressure: row.borrowing_pressure,
      wage_strength: row.wage_strength,
    });
  }
  const value = row[key];
  return typeof value === 'number' ? value : null;
}

function sparkData(scores: ScoreRow[], key: keyof ScoreRow | 'student_cost_pressure') {
  return scores
    .slice(-36)
    .map((row) => ({
      time: monthTimestamp(row.date),
      value: scoreValue(row, key),
    }))
    .filter((row) => row.value !== null) as KpiSparkPoint[];
}

function sparkDomain(series: KpiSparkPoint[]): [number, number] {
  const values = series.map((point) => point.value).filter((value): value is number => typeof value === 'number');
  if (!values.length) return [0, 100];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = Math.max(3, (max - min) * 0.2);
  return [Math.max(0, Math.floor(min - padding)), Math.min(100, Math.ceil(max + padding))];
}

function sparkTicks(series: KpiSparkPoint[]) {
  if (series.length < 2) return [];
  return [series[0].time, series[series.length - 1].time];
}

function healthForecastDescription(modelLabel: string | undefined, forecastDate: string | undefined, observed: number | null) {
  return `${modelLabel ?? 'Selected model'} sees the economy at risk of struggling by ${forecastDate ?? 'the next horizon'}, with recession-like pressure starting to show. Latest observed health is ${fmt(observed)}.`;
}

function useData() {
  const [data, setData] = useState<LoadedData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all(
      Object.entries(dataFiles).map(async ([key, path]) => {
        const response = await fetch(`${path}?v=${dataVersion}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Unable to load ${path}`);
        return [key, await response.json()];
      }),
    )
      .then((entries) => setData(Object.fromEntries(entries) as LoadedData))
      .catch((err) => setError(err.message));
  }, []);

  return { data, error };
}

function Card({
  title,
  value,
  detail,
  icon,
  tone = 'neutral',
}: {
  title: string;
  value: string;
  detail: string;
  icon: JSX.Element;
  tone?: string;
}) {
  return (
    <article className={`card card-${tone}`}>
      <div className="card-top">
        <span className="icon-chip">{icon}</span>
        <span className="card-title">{title}</span>
      </div>
      <div className="card-value">{value}</div>
      <p>{detail}</p>
    </article>
  );
}

function MetricCard({
  title,
  value,
  trend,
  icon,
  description,
  series,
  inverse = false,
  statusLabel,
  statusTone,
}: {
  title: string;
  value: number | null | undefined;
  trend?: number | null;
  icon: JSX.Element;
  description: string;
  series: KpiSparkPoint[];
  inverse?: boolean;
  statusLabel?: string | null;
  statusTone?: string;
}) {
  const band = statusLabel ? { label: statusLabel, tone: statusTone ?? scoreTone(value) } : metricBand(value, inverse);
  const tone = trendTone(trend, inverse);
  const TrendIcon = trend !== undefined && trend !== null && trend < 0 ? TrendingDown : TrendingUp;
  const yDomain = sparkDomain(series);
  const ticks = sparkTicks(series);
  const barColorValue = inverse ? 100 - Number(value ?? 0) : Number(value ?? 0);

  return (
    <article className="metric-card">
      <div className="metric-top">
        <span className="metric-icon">{icon}</span>
        <span className={`metric-trend trend-${tone}`}>
          <TrendIcon />
          {trendLabel(trend)}
        </span>
      </div>
      <h3>{title}</h3>
      <div className="metric-value">
        <span>{fmt(value)}</span>
        <small>/ 100</small>
      </div>
      <div className="metric-bar" aria-hidden="true">
        <span style={{ width: `${clamp(Number(value ?? 0))}%`, background: scoreFill(barColorValue) }} />
      </div>
      <div className="metric-spark" aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 8, right: 4, bottom: 18, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e6edf2" />
            <XAxis
              dataKey="time"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              ticks={ticks}
              tickFormatter={formatYearTick}
              tick={{ fontSize: 10, fill: '#7b8b96' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={yDomain}
              ticks={yDomain}
              tick={{ fontSize: 10, fill: '#7b8b96' }}
              axisLine={false}
              tickLine={false}
              width={28}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#2f6f9f"
              strokeWidth={2}
              dot={(props) => {
                const payload = props.payload as KpiSparkPoint;
                if (!payload?.forecast) return <g />;
                return <circle cx={props.cx} cy={props.cy} r={4} fill="#ff6b00" stroke="#ffffff" strokeWidth={2} />;
              }}
              activeDot={{ r: 4 }}
            />
            <Tooltip
              formatter={(tooltipValue) => [fmt(Number(tooltipValue)), title]}
              labelFormatter={(label) => formatMonthLabel(Number(label))}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <span className={`status-pill status-${band.tone}`}>{band.label}</span>
      <p>{description}</p>
    </article>
  );
}

function Dashboard({ data }: { data: LoadedData }) {
  const { dashboard, scores } = data;
  const dashboardModels: ArtifactModel[] =
    dashboard.overall_health_prediction?.models?.map((model: any) => ({
      key: model.key,
      label: model.label,
      status: model.status,
      prediction: model.prediction_3_months_ahead,
      metrics: model.metrics,
    })) ?? data.artifacts.models ?? [];
  const defaultModel =
    dashboard.overall_health_prediction?.best_model ?? dashboardModels.find((model) => model.status === 'available')?.key ?? '';
  const [selectedModelKey, setSelectedModelKey] = useState(defaultModel);
  const selectedModel =
    dashboardModels.find((model) => model.key === selectedModelKey) ??
    dashboardModels.find((model) => model.status === 'available');
  const modelHealthPrediction =
    selectedModel?.prediction ?? dashboard.overall_health_prediction?.prediction_3_months_ahead ?? dashboard.overall_health;
  const modelTrend =
    modelHealthPrediction !== null && modelHealthPrediction !== undefined && dashboard.overall_health !== null
      ? Number(modelHealthPrediction) - Number(dashboard.overall_health)
      : dashboard.trends.overall_health_change_1m;
  const explanation = dashboard.score_explanation ?? {
    summary: 'Score explanation is unavailable for this run.',
    drivers: [],
    supports: [],
    outlook: '',
    method: '',
  };
  const chartData = scores
    .filter((row) => row.overall_health !== null)
    .slice(-96)
    .map((row) => ({
      date: row.date.slice(0, 7),
      time: monthTimestamp(row.date),
      overall: row.overall_health,
      pressure: row.macro_state_label,
    }));
  const firstTime = chartData[0]?.time;
  const lastTime = chartData.at(-1)?.time;
  const firstTickYear =
    firstTime === undefined
      ? 0
      : new Date(firstTime).getUTCFullYear() + (new Date(firstTime).getUTCMonth() > 0 ? 1 : 0);
  const lastTickYear = lastTime === undefined ? -1 : new Date(lastTime).getUTCFullYear();
  const yearTicks =
    firstTime === undefined || lastTime === undefined
      ? []
      : Array.from({ length: Math.max(0, lastTickYear - firstTickYear + 1) }, (_, index) =>
          Date.UTC(firstTickYear + index, 0, 1),
        ).filter((tick) => tick >= firstTime && tick <= lastTime);

  const metricCards = [
    {
      title: 'Overall Economic Health',
      value: modelHealthPrediction,
      trend: modelTrend,
      icon: <Activity />,
      statusLabel: metricBand(modelHealthPrediction).label,
      statusTone: scoreTone(modelHealthPrediction),
      series: [
        ...sparkData(scores, 'overall_health'),
        ...(dashboard.overall_health_prediction?.prediction_date
          ? [
              {
                time: monthTimestamp(dashboard.overall_health_prediction.prediction_date),
                value: modelHealthPrediction,
                forecast: true,
              },
            ]
          : []),
      ],
      description: healthForecastDescription(
        selectedModel?.label,
        dashboard.overall_health_prediction?.prediction_date,
        dashboard.overall_health,
      ),
    },
    {
      title: 'Student Cost Pressure',
      value: dashboard.student_cost_pressure.current,
      trend: dashboard.trends.student_cost_pressure_change_1m,
      icon: <DollarSign />,
      inverse: true,
      statusLabel: pressureBand(dashboard.student_cost_pressure.current).label,
      statusTone: pressureBand(dashboard.student_cost_pressure.current).tone,
      series: sparkData(scores, 'student_cost_pressure'),
      description: 'Lower is better. This is a cost-pressure index, so higher values mean student budgets are under more strain.',
    },
    {
      title: 'Rent Pressure',
      value: dashboard.category_scores.rent_pressure,
      trend: latestChange(scores, 'rent_pressure'),
      icon: <Home />,
      series: sparkData(scores, 'rent_pressure'),
      description: 'Higher means housing costs are easier to absorb. Lower means rent and mortgage pressure is heavier.',
    },
    {
      title: 'Job Market Strength',
      value: dashboard.category_scores.job_market_strength,
      trend: latestChange(scores, 'job_market_strength'),
      icon: <Briefcase />,
      series: sparkData(scores, 'job_market_strength'),
      description: 'Higher means stronger hiring, lower unemployment, and better labor demand.',
    },
    {
      title: 'Inflation Pressure',
      value: dashboard.category_scores.inflation_pressure,
      trend: latestChange(scores, 'inflation_pressure'),
      icon: <ShoppingCart />,
      series: sparkData(scores, 'inflation_pressure'),
      description: 'Higher means prices are more stable. Lower means inflation is weighing more on households.',
    },
    {
      title: 'Borrowing Pressure',
      value: dashboard.category_scores.borrowing_pressure,
      trend: latestChange(scores, 'borrowing_pressure'),
      icon: <CreditCard />,
      series: sparkData(scores, 'borrowing_pressure'),
      description: 'Higher means credit is less restrictive. Lower means rates and debt costs are harder to manage.',
    },
    {
      title: 'Wage Strength',
      value: dashboard.category_scores.wage_strength,
      trend: latestChange(scores, 'wage_strength'),
      icon: <Wallet />,
      series: sparkData(scores, 'wage_strength'),
      description: 'Higher means wages are doing more to offset costs and support affordability.',
    },
    {
      title: 'Consumer Sentiment',
      value: dashboard.category_scores.consumer_sentiment,
      trend: latestChange(scores, 'consumer_sentiment'),
      icon: <Smile />,
      series: sparkData(scores, 'consumer_sentiment'),
      description: 'Higher means households feel more confident. Lower signals caution and financial stress.',
    },
  ];

  return (
    <div className="stack">
      <section className="project-overview">
        <div>
          <p className="eyebrow">HackDavis 2026 project</p>
          <h2>EconPulse makes macroeconomic health easier to understand.</h2>
          <p>
            EconPulse turns public FRED economic data into month-by-month scores for overall
            economic health, student cost pressure, jobs, rent, inflation, borrowing, wages, and
            consumer sentiment.
          </p>
        </div>
        <div className="project-goal">
          <strong>Goal</strong>
          <p>
            Help students and everyday users quickly see whether the economy feels stable,
            strained, or improving, and understand which conditions are driving that signal.
          </p>
        </div>
      </section>

      <section className="panel model-kpi-panel">
        <div className="section-head">
          <div>
            <p className="eyebrow">Dashboard model</p>
            <h2>Pick the model driving the health KPI</h2>
            <p>Only the overall economic health prediction changes by model. The other KPIs stay fixed to the latest observed scores.</p>
          </div>
          <Gauge />
        </div>
        <div className="model-selector">
          {dashboardModels.map((model) => (
            <button
              key={model.key}
              className={selectedModel?.key === model.key ? 'active' : ''}
              onClick={() => setSelectedModelKey(model.key)}
            >
              {model.label}
            </button>
          ))}
        </div>
      </section>

      <section className="dashboard-metrics" aria-label="Latest EconPulse scores">
        {metricCards.map((metric) => (
          <MetricCard key={metric.title} {...metric} />
        ))}
      </section>

      <section className="panel">
        <div className="section-head">
          <div>
            <p className="eyebrow">Timeline</p>
            <h2>Economic health over the years</h2>
          </div>
          <LineChartIcon />
        </div>
        <div className="chart tall">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="healthGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4f8fbf" stopOpacity={0.45} />
                  <stop offset="95%" stopColor="#4f8fbf" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#dde5ea" />
              <XAxis
                dataKey="time"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                ticks={yearTicks}
                tickFormatter={formatYearTick}
                interval={0}
              />
              <YAxis domain={[0, 100]} />
              <Tooltip labelFormatter={(label) => formatMonthLabel(Number(label))} />
              <Area dataKey="overall" stroke="#2f6f9f" fill="url(#healthGradient)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}

function ModelHub({ data }: { data: LoadedData }) {
  const models: ArtifactModel[] = data.artifacts.models ?? [];
  const [selected, setSelected] = useState(models[0]?.key ?? '');
  const selectedModel = models.find((model) => model.key === selected) ?? models[0];

  const comparison = models
    .filter((model) => model.metrics)
    .map((model) => ({
      name: model.label,
      rmse: model.metrics?.rmse,
      r2: model.metrics?.r2,
    }));

  return (
    <div className="stack">
      <section className="panel">
        <div className="section-head">
          <div>
            <p className="eyebrow">Model hub</p>
            <h2>Forecast model comparison</h2>
            <p>
              These saved .pkl models forecast overall economic health three months ahead using
              the six EconPulse category scores. Higher predicted health is better.
            </p>
          </div>
          <BrainCircuit />
        </div>

        <div className="model-selector">
          {models.map((model) => (
            <button
              key={model.key}
              className={selected === model.key ? 'active' : ''}
              onClick={() => setSelected(model.key)}
            >
              {model.label}
            </button>
          ))}
        </div>

        {selectedModel && (
          <div className="model-detail">
            <div>
              <p className="eyebrow">Selected model</p>
              <h3>{selectedModel.label}</h3>
              <p>
                {artifactDescriptions[selectedModel.key] ?? selectedModel.reason ?? selectedModel.model_class}
              </p>
            </div>
            <div className="metric-grid">
              <Card
                title="Forecasted Health"
                value={fmt(selectedModel.prediction)}
                detail="3-month forecast target, not observed data"
                icon={<TrendingUp />}
                tone="neutral"
              />
              <Card
                title="MAE"
                value={fmt(selectedModel.metrics?.mae, 2)}
                detail="Average miss in score points. Lower means the model is usually closer."
                icon={<Gauge />}
              />
              <Card
                title="RMSE"
                value={fmt(selectedModel.metrics?.rmse, 2)}
                detail="Penalizes large misses more than MAE. Lower means fewer big forecast errors."
                icon={<Gauge />}
              />
              <Card
                title="R2"
                value={fmt(selectedModel.metrics?.r2, 3)}
                detail="How much variation the model explains. Higher is better; below 0 means weak validation."
                icon={<BarChart3 />}
              />
            </div>
          </div>
        )}
      </section>

      <section className="grid two">
        <div className="panel">
          <div className="section-head">
            <div>
              <p className="eyebrow">Accuracy</p>
              <h2>RMSE by model</h2>
            </div>
          </div>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dde5ea" />
                <XAxis dataKey="name" interval={0} angle={-18} textAnchor="end" height={70} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="rmse" fill="#88a8bd" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel">
          <div className="section-head">
            <div>
              <p className="eyebrow">Model library</p>
              <h2>Available forecasting models</h2>
            </div>
          </div>
          <div className="artifact-list">
            {(data.artifacts.models as ArtifactModel[]).map((model) => (
              <div className="artifact-row" key={model.key}>
                <div>
                  <strong>{model.label}</strong>
                  <span>{artifactDescriptions[model.key] ?? model.reason ?? model.status}</span>
                </div>
                <b>{model.status === 'available' ? fmt(model.prediction) : model.status}</b>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function Simulator({ data }: { data: LoadedData }) {
  const availableModels = (data.artifacts.models as ArtifactModel[]).filter(
    (model) => model.status === 'available' && model.sensitivity,
  );
  const [selected, setSelected] = useState(availableModels[0]?.key ?? '');
  const selectedModel = availableModels.find((model) => model.key === selected) ?? availableModels[0];
  const [values, setValues] = useState<Record<string, number>>(data.artifacts.base_input);

  const exactMacroHealth = useMemo(() => calculateMacroHealth(values), [values]);
  const exactStudentPressure = useMemo(() => calculateStudentCostPressure(values), [values]);
  const maxMoveFromBaseline = useMemo(
    () =>
      Math.max(
        ...Object.entries(values).map(([feature, value]) =>
          Math.abs(value - (data.artifacts.base_input[feature] ?? value)),
        ),
      ),
    [data.artifacts.base_input, values],
  );

  const localArtifactEstimate = useMemo(() => {
    if (!selectedModel?.sensitivity) return selectedModel?.prediction ?? null;
    const base = selectedModel.prediction ?? 0;
    const estimate = Object.entries(values).reduce((total, [feature, value]) => {
      const sensitivity = selectedModel.sensitivity?.[feature];
      const baseValue = data.artifacts.base_input[feature] ?? value;
      return total + (sensitivity?.approx_slope_per_point ?? 0) * (value - baseValue);
    }, base);
    return clamp(estimate);
  }, [data.artifacts.base_input, selectedModel, values]);

  const estimateReliability =
    maxMoveFromBaseline <= 20
      ? 'This estimate is near the latest economy baseline, where model sensitivity is most reliable.'
      : 'Large slider moves are farther from the training baseline, so treat the model estimate as directional.';

  return (
    <div className="stack">
      <section className="panel">
        <div className="section-head">
          <div>
            <p className="eyebrow">What-if simulator</p>
            <h2>Move category scores and compare model response</h2>
            <p>
              Each slider is a 0-100 health score. Higher means better conditions for students and
              consumers; lower means more economic pressure.
            </p>
          </div>
          <SlidersHorizontal />
        </div>

        <div className="model-selector">
          {availableModels.map((model) => (
            <button
              key={model.key}
              className={selectedModel?.key === model.key ? 'active' : ''}
              onClick={() => setSelected(model.key)}
            >
              {model.label}
            </button>
          ))}
        </div>

        <div className="simulator-layout">
          <div className="slider-stack">
            {data.artifacts.features.map((feature: string) => (
              <label className="slider-row" key={feature}>
                <div className="slider-copy">
                  <span>
                    <strong>{categoryLabels[feature] ?? feature}</strong>
                    <small>{fmt(values[feature])}</small>
                  </span>
                  <p>{featureDescriptions[feature]}</p>
                  <em>{sliderEffects[feature]}</em>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="0.1"
                  value={values[feature]}
                  onChange={(event) =>
                    setValues((current) => ({ ...current, [feature]: Number(event.target.value) }))
                  }
                />
              </label>
            ))}
          </div>

          <div className="sim-output">
            <p className="eyebrow">Selected model response</p>
            <h3>{selectedModel?.label ?? 'Model estimate'}</h3>
            <div className="scenario-metrics">
              <div className="scenario-primary">
                <span>Model Forecast</span>
                <strong>{fmt(localArtifactEstimate)}</strong>
              </div>
              <div>
                <span>Formula Macro Health</span>
                <strong>{fmt(exactMacroHealth)}</strong>
              </div>
              <div>
                <span>Formula Cost Pressure</span>
                <strong>{fmt(exactStudentPressure)}</strong>
              </div>
            </div>
            <p>
              Switching models changes the model forecast because each saved model learned a
              different response to the six category scores. The formula values are a consistent
              baseline and do not change by model.
            </p>
            <details className="artifact-estimate">
              <summary>How to read this estimate</summary>
              <p className="eyebrow">Model caveat</p>
              <strong>{selectedModel?.label}</strong>
              <span>
                Baseline: {fmt(selectedModel?.prediction)} · scenario estimate:{' '}
                {fmt(localArtifactEstimate)}
              </span>
              <span>{estimateReliability}</span>
            </details>
            <p>
              Setting every input to 0 still makes formula macro health collapse toward 0 while
              formula student cost pressure rises toward 100.
            </p>
            <button className="secondary" onClick={() => setValues(data.artifacts.base_input)}>
              Reset to latest economy
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>('dashboard');
  const { data, error } = useData();

  if (error) {
    return <main className="app-shell">Could not load dashboard data: {error}</main>;
  }

  if (!data) {
    return <main className="app-shell">Loading EconPulse data...</main>;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">FRED data plus model validation</p>
          <h1>EconPulse</h1>
        </div>
        <div className="top-actions">
          <nav>
            <button className={tab === 'dashboard' ? 'active' : ''} onClick={() => setTab('dashboard')}>
              Dashboard
            </button>
            <button className={tab === 'models' ? 'active' : ''} onClick={() => setTab('models')}>
              Model Hub
            </button>
            <button className={tab === 'simulator' ? 'active' : ''} onClick={() => setTab('simulator')}>
              What-If
            </button>
          </nav>
        </div>
      </header>

      {tab === 'dashboard' && <Dashboard data={data} />}
      {tab === 'models' && <ModelHub data={data} />}
      {tab === 'simulator' && <Simulator data={data} />}
    </main>
  );
}
