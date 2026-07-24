import { Checkpoint, MinerData } from "../../types";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./powerBiDashboard.css";

interface PowerBiDashboardProps {
  data: MinerData;
}

type CheckpointSnapshot = Checkpoint & {
  last_update_ms?: number;
  overall_returns?: number;
};

type SalesPoint = {
  label: string;
  value: number;
};

type SegmentRow = {
  category: string;
  spend: number;
  savings: number;
  yoy: string;
};

const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"];
const ENGINE_PROGRAMS = ["LEAP-1B", "LEAP-1A", "GenX", "CFM-56"];

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const formatCompactCurrency = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);

const formatCurrency = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);

const toCheckpointSnapshots = (checkpoints: Checkpoint[]) =>
  checkpoints as CheckpointSnapshot[];

const buildMonthlyPurchasingSpend = (base: number) =>
  MONTH_LABELS.map((label, index) => ({
    label,
    value: Math.round(base * (0.9 + index * 0.09)),
  }));

const buildRepairSavings = (base: number) =>
  MONTH_LABELS.map((label, index) => ({
    label,
    value: Math.round(base * (0.62 + index * 0.07)),
  }));

const buildProgramSavings = (base: number) =>
  ENGINE_PROGRAMS.map((program, index) => ({
    label: program,
    value: Math.round(base * (1.08 - index * 0.15)),
  }));

export const PowerBiDashboard = ({ data }: PowerBiDashboardProps) => {
  const minerStats = data.statistics.data[0];
  const minerPositions = data.positions[minerStats.hotkey];
  const checkpointSnapshots = toCheckpointSnapshots(minerStats.checkpoints);

  const spendBase = Math.max(
    550000,
    Math.round(Math.abs(minerPositions.all_time_returns) * 380000 + minerPositions.n_positions * 21000),
  );
  const savingsBase = Math.max(260000, Math.round(spendBase * 0.28));

  const monthlyPurchasingSpend = buildMonthlyPurchasingSpend(spendBase);
  const monthlyRepairSavings = buildRepairSavings(savingsBase);
  const programSavings = buildProgramSavings(savingsBase);
  const sparklineSeries: SalesPoint[] =
    checkpointSnapshots.length > 0
      ? checkpointSnapshots.slice(-18).map((snapshot, index) => ({
          label: `${index + 1}`,
          value: Math.round(savingsBase * (0.48 + clamp(Number(snapshot.overall_returns ?? 1), 0.35, 1.65))),
        }))
      : monthlyRepairSavings;

  const totalPurchasingSpend = monthlyPurchasingSpend.reduce((sum, point) => sum + point.value, 0);
  const totalRepairSavings = monthlyRepairSavings.reduce((sum, point) => sum + point.value, 0);
  const outsideVendorRepairSpend = Math.round(totalPurchasingSpend * 0.34);
  const savingsRate = clamp(totalRepairSavings / totalPurchasingSpend, 0.05, 0.45);

  const progressGoal = 4000000;
  const progressPercentile = clamp(minerStats.weight.percentile * 0.6 + 0.22, 0.1, 0.96);
  const progressValue = Math.round(progressGoal * progressPercentile);
  const progressData = [
    { name: "progress", value: progressValue },
    { name: "remaining", value: progressGoal - progressValue },
  ];

  const segmentRows: SegmentRow[] = [
    {
      category: "Supply chain purchasing",
      spend: totalPurchasingSpend,
      savings: Math.round(totalPurchasingSpend * 0.12),
      yoy: "+9.4%",
    },
    {
      category: "Outside vendor repairs",
      spend: outsideVendorRepairSpend,
      savings: totalRepairSavings,
      yoy: "+14.2%",
    },
    {
      category: "Inventory optimization",
      spend: Math.round(totalPurchasingSpend * 0.21),
      savings: Math.round(totalPurchasingSpend * 0.06),
      yoy: "+7.8%",
    },
  ];

  return (
    <div className="pbi-page">
      <div className="pbi-dashboard">
        <header className="pbi-header">
          <div className="pbi-brand">MTU Maintenance</div>
          <nav className="pbi-nav">
            <span>Purchasing</span>
            <span className="is-active">Progress</span>
            <span>Vendor Repairs</span>
            <span>Engine Programs</span>
            <span>Report</span>
          </nav>
        </header>

        <section className="pbi-grid">
          <article className="pbi-card pbi-card--left">
            <h3>Total Purchasing Spend Trend</h3>
            <p>Sample monthly spend across LEAP-1B, LEAP-1A, GenX, and CFM-56 programs.</p>
            <div className="pbi-chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={monthlyPurchasingSpend}>
                  <CartesianGrid stroke="#22385f" strokeOpacity={0.35} vertical={false} />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fill: "#8aa0cf", fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} tick={{ fill: "#8aa0cf", fontSize: 11 }} tickFormatter={formatCompactCurrency} />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Area type="monotone" dataKey="value" stroke="#5d90ff" fill="#3f6fce" fillOpacity={0.35} strokeWidth={2.2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="pbi-metrics">
              <div>
                <strong>{formatCompactCurrency(totalPurchasingSpend)}</strong>
                <span>Total purchasing spend</span>
              </div>
              <div>
                <strong>{formatCompactCurrency(outsideVendorRepairSpend)}</strong>
                <span>Outside vendor repair spend</span>
              </div>
            </div>

            <table className="pbi-table">
              <thead>
                <tr>
                  <th>Cost bucket</th>
                  <th>Spend</th>
                  <th>Savings</th>
                  <th>YoY</th>
                </tr>
              </thead>
              <tbody>
                {segmentRows.map((row) => (
                  <tr key={row.category}>
                    <td>{row.category}</td>
                    <td>{formatCurrency(row.spend)}</td>
                    <td>{formatCurrency(row.savings)}</td>
                    <td>{row.yoy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>

          <article className="pbi-card pbi-card--purchases">
            <h3>Purchasing Savings</h3>
            <strong className="pbi-kpi">{formatCompactCurrency(segmentRows[0].savings)}</strong>
            <span className="pbi-pill">{segmentRows[0].yoy}</span>
            <div className="pbi-mini-chart">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sparklineSeries}>
                  <Area type="monotone" dataKey="value" stroke="#69a1ff" fill="#3f6fce" fillOpacity={0.38} strokeWidth={1.8} />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </article>

          <article className="pbi-card pbi-card--progress">
            <h3>YTD Savings Goal</h3>
            <p>Supply chain + outside vendor repairs</p>
            <strong className="pbi-progress-value">{formatCompactCurrency(progressValue)}</strong>
            <span className="pbi-progress-goal">/ {formatCurrency(progressGoal)}</span>
            <div className="pbi-donut-chart">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={progressData}
                    dataKey="value"
                    innerRadius={62}
                    outerRadius={82}
                    startAngle={180}
                    endAngle={0}
                    stroke="none"
                  >
                    <Cell fill="#4e80ff" />
                    <Cell fill="#f0f3fb" />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </article>

          <article className="pbi-card pbi-card--weekly">
            <h3>Savings by Engine Program</h3>
            <p>Weighted sample savings from purchasing and external repair actions.</p>
            <div className="pbi-chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={programSavings} layout="vertical" margin={{ top: 8, right: 20, left: 0, bottom: 10 }}>
                  <XAxis type="number" tickFormatter={formatCompactCurrency} tick={{ fill: "#8aa0cf", fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="label" tickLine={false} axisLine={false} tick={{ fill: "#8aa0cf", fontSize: 11 }} />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Bar dataKey="value" fill="#4e80ff" radius={[2, 2, 2, 2]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="pbi-metrics">
              <div>
                <strong>{formatCompactCurrency(totalRepairSavings)}</strong>
                <span>Total repair savings</span>
              </div>
              <div>
                <strong>{(savingsRate * 100).toFixed(1)}%</strong>
                <span>Savings rate</span>
              </div>
            </div>

            <table className="pbi-table">
              <thead>
                <tr>
                  <th>Program</th>
                  <th>Savings</th>
                  <th>Mix</th>
                </tr>
              </thead>
              <tbody>
                {programSavings.map((row) => (
                  <tr key={`weekly-${row.label}`}>
                    <td>{row.label}</td>
                    <td>{formatCurrency(row.value)}</td>
                    <td>{((row.value / totalRepairSavings) * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>

          <article className="pbi-card pbi-card--combo">
            <h3>Outside Vendor Repair Savings</h3>
            <p>Monthly savings generated through third-party repair strategy optimization.</p>

            <div className="pbi-combo-top">
              <div>
                <span>Outside vendor savings</span>
                <strong>{formatCompactCurrency(totalRepairSavings)}</strong>
              </div>
              <p>Sample view for MTU Maintenance airline-engine operations.</p>
            </div>

            <div className="pbi-chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={monthlyRepairSavings} margin={{ top: 4, right: 6, left: 0, bottom: 10 }}>
                  <CartesianGrid stroke="#22385f" strokeOpacity={0.3} vertical={false} />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fill: "#8aa0cf", fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} tick={{ fill: "#8aa0cf", fontSize: 11 }} tickFormatter={formatCompactCurrency} />
                  <Tooltip formatter={(value: number) => formatCurrency(value)} />
                  <Bar dataKey="value" fill="#4e80ff" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="pbi-footer-stats">
              <div>
                <strong>{formatCompactCurrency(segmentRows[1].savings)}</strong>
                <span>Savings from repairs</span>
              </div>
              <div>
                <strong>{formatCompactCurrency(segmentRows[1].spend)}</strong>
                <span>Outside vendor spend</span>
              </div>
              <div>
                <strong>{ENGINE_PROGRAMS.length}</strong>
                <span>Engine programs</span>
              </div>
            </div>
          </article>
        </section>
      </div>
    </div>
  );
};
