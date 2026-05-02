import { useState, useEffect, useMemo, useCallback } from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter, ComposedChart } from "recharts";

const COLORS = {
  bg: "#0a0e1a",
  card: "#111827",
  cardHover: "#1a2236",
  border: "#1e293b",
  accent: "#3b82f6",
  accentGlow: "#3b82f630",
  green: "#10b981",
  greenDim: "#10b98140",
  red: "#ef4444",
  redDim: "#ef444440",
  yellow: "#f59e0b",
  yellowDim: "#f59e0b40",
  purple: "#8b5cf6",
  cyan: "#06b6d4",
  orange: "#f97316",
  text: "#e2e8f0",
  textMuted: "#94a3b8",
  textDim: "#64748b",
};

const RISK_GRADES = [
  { grade: "AAA", pd: 0.003, count: 52400, dr: 0.002, lgd: 0.22, color: "#10b981" },
  { grade: "AA", pd: 0.008, count: 58200, dr: 0.006, lgd: 0.28, color: "#34d399" },
  { grade: "A", pd: 0.015, count: 61500, dr: 0.013, lgd: 0.33, color: "#6ee7b7" },
  { grade: "BBB", pd: 0.025, count: 55800, dr: 0.022, lgd: 0.38, color: "#a7f3d0" },
  { grade: "BB+", pd: 0.038, count: 49200, dr: 0.035, lgd: 0.42, color: "#fbbf24" },
  { grade: "BB", pd: 0.055, count: 52100, dr: 0.050, lgd: 0.45, color: "#f59e0b" },
  { grade: "BB-", pd: 0.078, count: 47500, dr: 0.072, lgd: 0.48, color: "#f97316" },
  { grade: "B", pd: 0.115, count: 43800, dr: 0.108, lgd: 0.52, color: "#fb923c" },
  { grade: "B-", pd: 0.165, count: 41200, dr: 0.155, lgd: 0.58, color: "#ef4444" },
  { grade: "CCC", pd: 0.280, count: 38300, dr: 0.260, lgd: 0.65, color: "#dc2626" },
];

const ROC_DATA = Array.from({ length: 51 }, (_, i) => {
  const fpr = i / 50;
  const tpr = Math.min(1, Math.pow(fpr, 0.22));
  return { fpr: +(fpr * 100).toFixed(1), tpr: +(tpr * 100).toFixed(1), random: +(fpr * 100).toFixed(1) };
});

const VINTAGE_DATA = [
  { month: "M0", Q1_2023: 0, Q2_2023: 0, Q3_2023: 0, Q4_2023: 0, Q1_2024: 0 },
  { month: "M3", Q1_2023: 0.8, Q2_2023: 0.9, Q3_2023: 1.0, Q4_2023: 1.1, Q1_2024: 1.2 },
  { month: "M6", Q1_2023: 1.8, Q2_2023: 2.0, Q3_2023: 2.3, Q4_2023: 2.5, Q1_2024: 2.8 },
  { month: "M9", Q1_2023: 2.8, Q2_2023: 3.1, Q3_2023: 3.5, Q4_2023: 3.8, Q1_2024: 4.2 },
  { month: "M12", Q1_2023: 3.5, Q2_2023: 3.9, Q3_2023: 4.3, Q4_2023: 4.7, Q1_2024: null },
  { month: "M15", Q1_2023: 4.0, Q2_2023: 4.5, Q3_2023: 5.0, Q4_2023: null, Q1_2024: null },
  { month: "M18", Q1_2023: 4.3, Q2_2023: 4.9, Q3_2023: null, Q4_2023: null, Q1_2024: null },
];

const STRESS_DATA = [
  { scenario: "Baseline", pd: 4.5, lgd: 42, el: 580, rwa: 4200, pdMigration: 0 },
  { scenario: "Adverse", pd: 6.5, lgd: 48, el: 960, rwa: 5800, pdMigration: 45 },
  { scenario: "Severely Adverse", pd: 9.8, lgd: 55, el: 1650, rwa: 7400, pdMigration: 118 },
];

const MIGRATION_MATRIX = [
  [92.1, 6.2, 1.2, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01, 0.01],
  [3.1, 88.5, 6.4, 1.3, 0.4, 0.15, 0.08, 0.04, 0.02, 0.01],
  [0.5, 3.8, 87.2, 6.5, 1.3, 0.4, 0.15, 0.08, 0.05, 0.02],
  [0.1, 0.6, 4.2, 86.8, 6.1, 1.4, 0.4, 0.2, 0.1, 0.1],
  [0.05, 0.15, 0.7, 4.5, 85.5, 6.8, 1.5, 0.4, 0.2, 0.2],
  [0.02, 0.08, 0.2, 0.8, 5.0, 84.2, 7.2, 1.6, 0.5, 0.38],
  [0.01, 0.04, 0.1, 0.3, 0.8, 5.5, 82.5, 7.8, 1.8, 1.16],
  [0.01, 0.02, 0.05, 0.15, 0.3, 1.0, 6.0, 80.0, 8.5, 3.97],
  [0.0, 0.01, 0.02, 0.08, 0.15, 0.4, 1.2, 6.5, 78.0, 13.64],
  [0.0, 0.0, 0.01, 0.04, 0.08, 0.2, 0.5, 1.5, 7.0, 90.67],
];

const ROLL_RATE_DATA = [
  { status: "Current>30", rate: 3.2, color: COLORS.green },
  { status: "30>60", rate: 28.5, color: COLORS.yellow },
  { status: "60>90", rate: 42.1, color: COLORS.orange },
  { status: "90>120", rate: 55.8, color: COLORS.red },
  { status: "120>Default", rate: 72.3, color: "#991b1b" },
];

const PSI_DATA = Array.from({ length: 24 }, (_, i) => ({
  month: `M${i + 1}`,
  psi: +(0.02 + Math.sin(i * 0.3) * 0.03 + Math.random() * 0.02).toFixed(3),
}));

const FEATURE_IMPORTANCE = [
  { feature: "Credit Score", importance: 18.5, iv: 0.62 },
  { feature: "DTI Ratio", importance: 14.2, iv: 0.48 },
  { feature: "Credit Utilization", importance: 12.8, iv: 0.42 },
  { feature: "Payment History", importance: 10.5, iv: 0.38 },
  { feature: "Delinquencies (2Y)", importance: 8.9, iv: 0.35 },
  { feature: "Employment Years", importance: 7.2, iv: 0.28 },
  { feature: "LTV Ratio", importance: 6.5, iv: 0.24 },
  { feature: "Interest Rate", importance: 5.8, iv: 0.21 },
  { feature: "Loan Amount (log)", importance: 5.1, iv: 0.18 },
  { feature: "Macro Stress Index", importance: 4.3, iv: 0.15 },
  { feature: "Income (log)", importance: 3.6, iv: 0.12 },
  { feature: "Credit Depth", importance: 2.6, iv: 0.09 },
];

const tabs = [
  { id: "overview", label: "Overview", icon: ">>" },
  { id: "pd", label: "PD Model", icon: "::" },
  { id: "lgd_ead", label: "LGD / EAD", icon: "<>" },
  { id: "validation", label: "Validation", icon: "[]" },
  { id: "stress", label: "Stress Test", icon: "//" },
  { id: "capital", label: "Capital", icon: "##" },
  { id: "scorer", label: "Live Scorer", icon: "~>" },
];

function MetricCard({ label, value, sub, trend, color, glow }) {
  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: "18px 20px", position: "relative", overflow: "hidden" }}>
      {glow && <div style={{ position: "absolute", top: -20, right: -20, width: 80, height: 80, borderRadius: "50%", background: glow, filter: "blur(30px)", opacity: 0.4 }} />}
      <div style={{ fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 6, fontFamily: "'JetBrains Mono', monospace" }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || COLORS.text, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: COLORS.textDim, marginTop: 4 }}>{sub}</div>}
      {trend && <div style={{ fontSize: 11, color: trend > 0 ? COLORS.green : COLORS.red, marginTop: 4 }}>{trend > 0 ? "+" : "-"}{Math.abs(trend)}%</div>}
    </div>
  );
}

function SectionHeader({ title, subtitle }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: COLORS.text, margin: 0, fontFamily: "'Space Grotesk', sans-serif" }}>{title}</h2>
      {subtitle && <p style={{ fontSize: 13, color: COLORS.textDim, margin: "4px 0 0" }}>{subtitle}</p>}
    </div>
  );
}

function ChartCard({ title, children, span }) {
  return (
    <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: 20, gridColumn: span ? `span ${span}` : undefined }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textMuted, marginBottom: 16, textTransform: "uppercase", letterSpacing: 0.8 }}>{title}</div>
      {children}
    </div>
  );
}

function TrafficLight({ zone }) {
  const colors = { green: COLORS.green, yellow: COLORS.yellow, red: COLORS.red };
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      {["green", "yellow", "red"].map(z => (
        <div key={z} style={{ width: 14, height: 14, borderRadius: "50%", background: z === zone ? colors[z] : "#1e293b", border: `2px solid ${z === zone ? colors[z] : "#334155"}`, boxShadow: z === zone ? `0 0 8px ${colors[z]}60` : "none" }} />
      ))}
      <span style={{ fontSize: 12, color: colors[zone], marginLeft: 4, fontWeight: 600, textTransform: "uppercase" }}>{zone}</span>
    </div>
  );
}

function OverviewTab() {
  const portfolioBreakdown = [
    { name: "Mortgage", value: 25, color: COLORS.accent },
    { name: "Auto", value: 15, color: COLORS.green },
    { name: "Personal", value: 15, color: COLORS.purple },
    { name: "Credit Card", value: 15, color: COLORS.cyan },
    { name: "Small Business", value: 10, color: COLORS.orange },
    { name: "Student", value: 10, color: COLORS.yellow },
    { name: "Home Equity", value: 10, color: COLORS.red },
  ];

  return (
    <div>
      <SectionHeader title="Portfolio Overview" subtitle="Basel III Credit Risk Quantification | 500K+ Loan Applications" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 24 }}>
        <MetricCard label="Total Exposure" value="$2.3B" sub="500K+ accounts" glow={COLORS.accentGlow} />
        <MetricCard label="Default Rate" value="4.5%" sub="12-month window" color={COLORS.yellow} glow={COLORS.yellowDim} />
        <MetricCard label="Expected Loss" value="$43.6M" sub="EL = PD x LGD x EAD" color={COLORS.red} glow={COLORS.redDim} />
        <MetricCard label="Risk-Weighted Assets" value="$892M" sub="IRB Advanced" color={COLORS.purple} />
        <MetricCard label="Capital Required" value="$93.7M" sub="10.5% ratio" color={COLORS.cyan} />
        <MetricCard label="Economic Capital" value="$156M" sub="99.9% VaR" color={COLORS.orange} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
        <ChartCard title="Risk Grade Distribution and Default Rates">
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={RISK_GRADES}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="grade" tick={{ fill: COLORS.textMuted, fontSize: 11 }} />
              <YAxis yAxisId="left" tick={{ fill: COLORS.textMuted, fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fill: COLORS.textMuted, fontSize: 11 }} tickFormatter={v => `${v}%`} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
              <Bar yAxisId="left" dataKey="count" name="Account Count" radius={[4, 4, 0, 0]}>
                {RISK_GRADES.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.7} />)}
              </Bar>
              <Line yAxisId="right" type="monotone" dataKey="dr" name="Default Rate (%)" stroke={COLORS.red} strokeWidth={2.5} dot={{ r: 4, fill: COLORS.red }} />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Portfolio Breakdown">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={portfolioBreakdown} cx="50%" cy="50%" innerRadius={55} outerRadius={95} paddingAngle={3} dataKey="value">
                {portfolioBreakdown.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} formatter={v => `${v}%`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

function PDTab() {
  return (
    <div>
      <SectionHeader title="Probability of Default Model" subtitle="XGBoost + Logistic Regression | Vasicek TTC/PIT Calibration" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 24 }}>
        <MetricCard label="AUC" value="0.82" color={COLORS.green} glow={COLORS.greenDim} />
        <MetricCard label="Gini" value="0.64" color={COLORS.accent} />
        <MetricCard label="KS Statistic" value="0.45" color={COLORS.purple} />
        <MetricCard label="Brier Score" value="0.031" color={COLORS.cyan} />
        <MetricCard label="Grade Separation" value="15x" sub="Best vs Worst" color={COLORS.orange} />
        <MetricCard label="IV (Top Feature)" value="0.62" sub="Credit Score" color={COLORS.yellow} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <ChartCard title="ROC Curve (AUC = 0.82)">
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={ROC_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="fpr" tick={{ fill: COLORS.textMuted, fontSize: 10 }} label={{ value: "FPR (%)", position: "insideBottom", offset: -5, fill: COLORS.textDim, fontSize: 11 }} />
              <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} label={{ value: "TPR (%)", angle: -90, position: "insideLeft", fill: COLORS.textDim, fontSize: 11 }} />
              <Area type="monotone" dataKey="tpr" stroke={COLORS.accent} fill={COLORS.accentGlow} strokeWidth={2.5} name="Model" />
              <Line type="monotone" dataKey="random" stroke={COLORS.textDim} strokeDasharray="5 5" strokeWidth={1} name="Random" dot={false} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Feature Importance (Top 12)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tick={{ fill: COLORS.textMuted, fontSize: 10 }} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="feature" tick={{ fill: COLORS.textMuted, fontSize: 9 }} width={110} />
              <Bar dataKey="importance" name="Importance %" fill={COLORS.accent} radius={[0, 4, 4, 0]} barSize={14} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
      <ChartCard title="Information Value Analysis (WoE)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={FEATURE_IMPORTANCE}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="feature" tick={{ fill: COLORS.textMuted, fontSize: 9 }} angle={-25} textAnchor="end" height={60} />
            <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} />
            <Bar dataKey="iv" name="Information Value" radius={[4, 4, 0, 0]} barSize={28}>
              {FEATURE_IMPORTANCE.map((e, i) => (
                <Cell key={i} fill={e.iv > 0.3 ? COLORS.green : e.iv > 0.1 ? COLORS.yellow : COLORS.textDim} />
              ))}
            </Bar>
            <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ display: "flex", gap: 20, marginTop: 8, fontSize: 11, color: COLORS.textDim }}>
          <span style={{ color: COLORS.green }}>Strong (IV &gt; 0.3)</span>
          <span style={{ color: COLORS.yellow }}>Medium (0.1 to 0.3)</span>
          <span style={{ color: COLORS.textDim }}>Weak (&lt; 0.1)</span>
        </div>
      </ChartCard>
    </div>
  );
}

function LGDEADTab() {
  const lgdDist = Array.from({ length: 20 }, (_, i) => ({
    bin: `${(i * 5)}%`,
    count: Math.round(2000 * Math.exp(-0.5 * Math.pow((i * 5 - 42) / 15, 2))),
    predicted: Math.round(1800 * Math.exp(-0.5 * Math.pow((i * 5 - 40) / 16, 2))),
  }));
  const survivalData = Array.from({ length: 13 }, (_, i) => ({
    month: `M${i}`, low: +(100 * Math.exp(-0.01 * i)).toFixed(1), medium: +(100 * Math.exp(-0.03 * i)).toFixed(1), high: +(100 * Math.exp(-0.08 * i)).toFixed(1),
  }));

  return (
    <div>
      <SectionHeader title="LGD and EAD Models" subtitle="Beta Regression + Two-Stage GBM | Cox Regression + CCF Estimation" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 24 }}>
        <MetricCard label="LGD R-sq" value="0.71" color={COLORS.green} glow={COLORS.greenDim} />
        <MetricCard label="LGD RMSE" value="0.18" color={COLORS.accent} />
        <MetricCard label="Mean LGD" value="42%" sub="Defaulted accounts" color={COLORS.yellow} />
        <MetricCard label="EAD C-index" value="0.76" color={COLORS.purple} glow="#8b5cf640" />
        <MetricCard label="EAD MAPE" value="8.3%" color={COLORS.cyan} />
        <MetricCard label="CCF (Revolving)" value="0.65" sub="Avg. conversion" color={COLORS.orange} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <ChartCard title="LGD Distribution: Actual vs Predicted">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={lgdDist}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="bin" tick={{ fill: COLORS.textMuted, fontSize: 10 }} />
              <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} />
              <Bar dataKey="count" name="Actual" fill={COLORS.accent} fillOpacity={0.6} radius={[3, 3, 0, 0]} />
              <Bar dataKey="predicted" name="Predicted" fill={COLORS.purple} fillOpacity={0.6} radius={[3, 3, 0, 0]} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Survival Curves by Risk Segment">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={survivalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="month" tick={{ fill: COLORS.textMuted, fontSize: 10 }} />
              <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Line type="monotone" dataKey="low" name="Low Risk" stroke={COLORS.green} strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="medium" name="Medium Risk" stroke={COLORS.yellow} strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="high" name="High Risk" stroke={COLORS.red} strokeWidth={2.5} dot={false} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

function ValidationTab() {
  return (
    <div>
      <SectionHeader title="Model Validation and Monitoring" subtitle="Hosmer-Lemeshow, PSI, Back-Testing, Migration Analysis" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 24 }}>
        <MetricCard label="H-L Test chi-sq" value="12.4" sub="p = 0.19 (Pass)" color={COLORS.green} glow={COLORS.greenDim} />
        <MetricCard label="PSI" value="0.08" sub="Stable (< 0.10)" color={COLORS.green} />
        <MetricCard label="A/E Ratio" value="1.12" sub="3-year backtest" color={COLORS.accent} />
        <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: "18px 20px" }}>
          <div style={{ fontSize: 11, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 10, fontFamily: "'JetBrains Mono', monospace" }}>Traffic Light</div>
          <TrafficLight zone="green" />
          <div style={{ fontSize: 11, color: COLORS.textDim, marginTop: 8 }}>Binomial test p &lt; 0.05</div>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <ChartCard title="PSI Monitoring (24 Months)">
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart data={PSI_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="month" tick={{ fill: COLORS.textMuted, fontSize: 9 }} interval={2} />
              <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} domain={[0, 0.12]} />
              <Area type="monotone" dataKey="psi" fill={COLORS.accentGlow} stroke={COLORS.accent} strokeWidth={2} name="PSI" />
              <Line type="monotone" dataKey={() => 0.1} stroke={COLORS.yellow} strokeDasharray="5 5" strokeWidth={1} name="Warning" dot={false} />
              <Line type="monotone" dataKey={() => 0.25} stroke={COLORS.red} strokeDasharray="5 5" strokeWidth={1} name="Critical" dot={false} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Vintage Analysis: Cumulative Default Rate">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={VINTAGE_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="month" tick={{ fill: COLORS.textMuted, fontSize: 10 }} />
              <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} tickFormatter={v => `${v}%`} />
              <Line type="monotone" dataKey="Q1_2023" stroke={COLORS.accent} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
              <Line type="monotone" dataKey="Q2_2023" stroke={COLORS.purple} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
              <Line type="monotone" dataKey="Q3_2023" stroke={COLORS.cyan} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
              <Line type="monotone" dataKey="Q4_2023" stroke={COLORS.orange} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
              <Line type="monotone" dataKey="Q1_2024" stroke={COLORS.green} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <ChartCard title="Roll Rate Analysis">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={ROLL_RATE_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="status" tick={{ fill: COLORS.textMuted, fontSize: 10 }} />
              <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} tickFormatter={v => `${v}%`} />
              <Bar dataKey="rate" name="Roll Rate %" radius={[4, 4, 0, 0]} barSize={36}>
                {ROLL_RATE_DATA.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Bar>
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Rating Migration Matrix (1-Year)">
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}>
              <thead>
                <tr>
                  <th style={{ padding: "4px 6px", color: COLORS.textMuted, textAlign: "left", borderBottom: `1px solid ${COLORS.border}` }}>From\To</th>
                  {RISK_GRADES.map(g => <th key={g.grade} style={{ padding: "4px 3px", color: COLORS.textMuted, textAlign: "center", borderBottom: `1px solid ${COLORS.border}` }}>{g.grade}</th>)}
                </tr>
              </thead>
              <tbody>
                {RISK_GRADES.map((g, i) => (
                  <tr key={g.grade}>
                    <td style={{ padding: "3px 6px", color: COLORS.text, fontWeight: 600, borderBottom: `1px solid ${COLORS.border}` }}>{g.grade}</td>
                    {MIGRATION_MATRIX[i].map((v, j) => (
                      <td key={j} style={{ padding: "3px 3px", textAlign: "center", borderBottom: `1px solid ${COLORS.border}`, color: i === j ? COLORS.green : v > 5 ? COLORS.yellow : COLORS.textDim, fontWeight: i === j ? 700 : 400, background: i === j ? `${COLORS.green}10` : "transparent" }}>{v.toFixed(1)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ChartCard>
      </div>
    </div>
  );
}

function StressTab() {
  return (
    <div>
      <SectionHeader title="Stress Testing Framework" subtitle="CCAR/DFAST Scenarios: Unemployment, GDP, Property Value Shocks" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 24 }}>
        {STRESS_DATA.map((s, i) => (
          <div key={s.scenario} style={{ background: COLORS.card, border: `1px solid ${i === 0 ? COLORS.green : i === 1 ? COLORS.yellow : COLORS.red}40`, borderRadius: 12, padding: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: i === 0 ? COLORS.green : i === 1 ? COLORS.yellow : COLORS.red, marginBottom: 12 }}>{s.scenario}</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 12 }}>
              <div><span style={{ color: COLORS.textDim }}>PD:</span> <span style={{ color: COLORS.text, fontWeight: 600 }}>{s.pd}%</span></div>
              <div><span style={{ color: COLORS.textDim }}>LGD:</span> <span style={{ color: COLORS.text, fontWeight: 600 }}>{s.lgd}%</span></div>
              <div><span style={{ color: COLORS.textDim }}>EL:</span> <span style={{ color: COLORS.text, fontWeight: 600 }}>${s.el}M</span></div>
              <div><span style={{ color: COLORS.textDim }}>RWA:</span> <span style={{ color: COLORS.text, fontWeight: 600 }}>${s.rwa}M</span></div>
              <div style={{ gridColumn: "span 2" }}><span style={{ color: COLORS.textDim }}>PD Migration:</span> <span style={{ color: s.pdMigration > 50 ? COLORS.red : s.pdMigration > 0 ? COLORS.yellow : COLORS.green, fontWeight: 700 }}>+{s.pdMigration}%</span></div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <ChartCard title="Scenario Impact: Expected Loss ($M)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={STRESS_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="scenario" tick={{ fill: COLORS.textMuted, fontSize: 11 }} />
              <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} tickFormatter={v => `$${v}M`} />
              <Bar dataKey="el" name="Expected Loss ($M)" radius={[6, 6, 0, 0]} barSize={50}>
                <Cell fill={COLORS.green} />
                <Cell fill={COLORS.yellow} />
                <Cell fill={COLORS.red} />
              </Bar>
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Macro Scenario Parameters">
          <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
            {[
              { param: "Unemployment Rate", baseline: "5.5%", adverse: "+3pp -> 8.5%", severe: "+5pp -> 10.5%" },
              { param: "GDP Growth", baseline: "2.5%", adverse: "-2% -> 0.5%", severe: "-4% -> -1.5%" },
              { param: "Property Values", baseline: "n/a", adverse: "-20%", severe: "-35%" },
              { param: "Fed Funds Rate", baseline: "3.0%", adverse: "+1.5pp", severe: "+2.5pp" },
            ].map((r) => (
              <div key={r.param} style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr 0.8fr 0.8fr", gap: 8, fontSize: 12, padding: "8px 0", borderBottom: `1px solid ${COLORS.border}` }}>
                <span style={{ color: COLORS.text, fontWeight: 600 }}>{r.param}</span>
                <span style={{ color: COLORS.green }}>{r.baseline}</span>
                <span style={{ color: COLORS.yellow }}>{r.adverse}</span>
                <span style={{ color: COLORS.red }}>{r.severe}</span>
              </div>
            ))}
          </div>
        </ChartCard>
      </div>
    </div>
  );
}

function CapitalTab() {
  const capitalData = [
    { component: "Minimum Capital", value: 8.0, amount: 71.4, color: COLORS.accent },
    { component: "Conservation Buffer", value: 2.5, amount: 22.3, color: COLORS.purple },
    { component: "Countercyclical", value: 0.5, amount: 4.5, color: COLORS.cyan },
    { component: "D-SIB Surcharge", value: 1.0, amount: 8.9, color: COLORS.orange },
  ];

  return (
    <div>
      <SectionHeader title="Regulatory Capital Allocation" subtitle="Basel III IRB Advanced | RWA, Economic Capital, EL Decomposition" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 24 }}>
        <MetricCard label="Total RWA" value="$892M" sub="IRB Advanced" color={COLORS.accent} glow={COLORS.accentGlow} />
        <MetricCard label="RWA Density" value="38.8%" sub="RWA / Exposure" color={COLORS.purple} />
        <MetricCard label="Min. Capital (8%)" value="$71.4M" color={COLORS.green} />
        <MetricCard label="Total Requirement" value="$107.0M" sub="12% incl. buffers" color={COLORS.red} />
        <MetricCard label="EC (99.9%)" value="$156M" sub="Economic Capital" color={COLORS.orange} />
        <MetricCard label="RWA Accuracy" value="+/-2%" sub="vs Regulatory Filing" color={COLORS.cyan} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <ChartCard title="Capital Stack Composition">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={capitalData} layout="vertical" margin={{ left: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" tick={{ fill: COLORS.textMuted, fontSize: 10 }} tickFormatter={v => `$${v}M`} />
              <YAxis type="category" dataKey="component" tick={{ fill: COLORS.textMuted, fontSize: 11 }} width={140} />
              <Bar dataKey="amount" name="Capital ($M)" radius={[0, 6, 6, 0]} barSize={24}>
                {capitalData.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Bar>
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="EL Decomposition by Risk Grade">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={RISK_GRADES.map(g => ({ ...g, el: +(g.pd * g.lgd * g.count * 35 / 1e6).toFixed(1) }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="grade" tick={{ fill: COLORS.textMuted, fontSize: 11 }} />
              <YAxis tick={{ fill: COLORS.textMuted, fontSize: 10 }} tickFormatter={v => `$${v}M`} />
              <Bar dataKey="el" name="Expected Loss ($M)" radius={[4, 4, 0, 0]}>
                {RISK_GRADES.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Bar>
              <Tooltip contentStyle={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, fontSize: 12 }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

function LiveScorer() {
  const [inputs, setInputs] = useState({
    creditScore: 680,
    dti: 0.35,
    utilization: 0.45,
    loanAmount: 150000,
    ltv: 0.80,
    employmentYears: 5,
    delinquencies: 0,
    income: 75000,
  });

  const score = useMemo(() => {
    const logit = -3.5
      - 0.03 * (inputs.creditScore - 600)
      + 0.8 * inputs.dti
      + 1.2 * inputs.utilization
      + 0.15 * inputs.delinquencies
      - 0.02 * inputs.employmentYears
      + 0.3 * Math.max(inputs.ltv - 0.8, 0);
    const pd = 1 / (1 + Math.exp(-logit));
    const lgd = inputs.ltv > 0.3 ? 0.25 + 0.15 * inputs.ltv : 0.45;
    const ead = inputs.loanAmount * (1 + 0.1 * inputs.utilization);
    const el = pd * lgd * ead;
    const grade = pd < 0.005 ? "AAA" : pd < 0.01 ? "AA" : pd < 0.02 ? "A" : pd < 0.03 ? "BBB" : pd < 0.05 ? "BB+" : pd < 0.07 ? "BB" : pd < 0.10 ? "BB-" : pd < 0.15 ? "B" : pd < 0.20 ? "B-" : "CCC";
    const rho = 0.12 * (1 - Math.exp(-50 * pd)) / (1 - Math.exp(-50)) + 0.24 * (1 - (1 - Math.exp(-50 * pd)) / (1 - Math.exp(-50)));
    return { pd, lgd, ead, el, grade, rho };
  }, [inputs]);

  const sliders = [
    { key: "creditScore", label: "Credit Score", min: 300, max: 850, step: 10 },
    { key: "dti", label: "DTI Ratio", min: 0, max: 2, step: 0.05 },
    { key: "utilization", label: "Credit Utilization", min: 0, max: 1, step: 0.05 },
    { key: "loanAmount", label: "Loan Amount ($)", min: 5000, max: 500000, step: 5000 },
    { key: "ltv", label: "LTV Ratio", min: 0, max: 1.5, step: 0.05 },
    { key: "employmentYears", label: "Employment (Years)", min: 0, max: 40, step: 1 },
    { key: "delinquencies", label: "Delinquencies (2Y)", min: 0, max: 10, step: 1 },
    { key: "income", label: "Annual Income ($)", min: 15000, max: 500000, step: 5000 },
  ];

  const gradeColor = score.pd < 0.02 ? COLORS.green : score.pd < 0.05 ? COLORS.yellow : score.pd < 0.10 ? COLORS.orange : COLORS.red;

  return (
    <div>
      <SectionHeader title="Live Credit Risk Scorer" subtitle="Interactive PD/LGD/EAD calculation with real-time risk grading" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: 24 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textMuted, marginBottom: 16, textTransform: "uppercase", letterSpacing: 0.8 }}>Borrower Parameters</div>
          {sliders.map(s => (
            <div key={s.key} style={{ marginBottom: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <label style={{ fontSize: 12, color: COLORS.textMuted }}>{s.label}</label>
                <span style={{ fontSize: 12, color: COLORS.accent, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                  {s.key === "loanAmount" || s.key === "income" ? `$${inputs[s.key].toLocaleString()}` : inputs[s.key]}
                </span>
              </div>
              <input type="range" min={s.min} max={s.max} step={s.step} value={inputs[s.key]}
                onChange={e => setInputs(p => ({ ...p, [s.key]: +e.target.value }))}
                style={{ width: "100%", accentColor: COLORS.accent, height: 4 }} />
            </div>
          ))}
        </div>
        <div>
          <div style={{ background: `linear-gradient(135deg, ${gradeColor}15, ${COLORS.card})`, border: `1px solid ${gradeColor}40`, borderRadius: 12, padding: 24, marginBottom: 16, textAlign: "center" }}>
            <div style={{ fontSize: 11, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 1.5 }}>Risk Grade</div>
            <div style={{ fontSize: 52, fontWeight: 800, color: gradeColor, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.2, textShadow: `0 0 30px ${gradeColor}40` }}>{score.grade}</div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <MetricCard label="Probability of Default" value={`${(score.pd * 100).toFixed(3)}%`} color={gradeColor} />
            <MetricCard label="Loss Given Default" value={`${(score.lgd * 100).toFixed(1)}%`} color={COLORS.purple} />
            <MetricCard label="Exposure at Default" value={`$${Math.round(score.ead).toLocaleString()}`} color={COLORS.cyan} />
            <MetricCard label="Expected Loss" value={`$${Math.round(score.el).toLocaleString()}`} color={COLORS.red} glow={COLORS.redDim} />
            <MetricCard label="Asset Correlation (rho)" value={score.rho.toFixed(4)} sub="Vasicek model" color={COLORS.orange} />
            <MetricCard label="Risk Weight" value={`${(score.pd < 0.01 ? 20 : score.pd < 0.03 ? 50 : score.pd < 0.10 ? 100 : 150)}%`} sub="IRB standardized" color={COLORS.accent} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function CreditRiskDashboard() {
  const [activeTab, setActiveTab] = useState("overview");

  const renderTab = () => {
    switch (activeTab) {
      case "overview": return <OverviewTab />;
      case "pd": return <PDTab />;
      case "lgd_ead": return <LGDEADTab />;
      case "validation": return <ValidationTab />;
      case "stress": return <StressTab />;
      case "capital": return <CapitalTab />;
      case "scorer": return <LiveScorer />;
      default: return <OverviewTab />;
    }
  };

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
      
      {/* Header */}
      <div style={{ background: "linear-gradient(180deg, #111827 0%, #0a0e1a 100%)", borderBottom: `1px solid ${COLORS.border}`, padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 36, height: 36, borderRadius: 8, background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.purple})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, fontWeight: 800 }}>CR</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: -0.3 }}>Credit Risk Platform</div>
            <div style={{ fontSize: 11, color: COLORS.textDim }}>PD/LGD/EAD | Basel III IRB Advanced</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: COLORS.green, boxShadow: `0 0 6px ${COLORS.green}` }} />
          <span style={{ fontSize: 11, color: COLORS.green }}>Pipeline Active</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 2, padding: "8px 24px", background: COLORS.card, borderBottom: `1px solid ${COLORS.border}`, overflowX: "auto" }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            style={{ padding: "8px 16px", borderRadius: 8, border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600, whiteSpace: "nowrap",
              background: activeTab === t.id ? `${COLORS.accent}20` : "transparent",
              color: activeTab === t.id ? COLORS.accent : COLORS.textDim,
              transition: "all 0.15s" }}>
            <span style={{ marginRight: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}>{t.icon}</span>{t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: 24, maxWidth: 1200, margin: "0 auto" }}>
        {renderTab()}
      </div>

      {/* Footer */}
      <div style={{ textAlign: "center", padding: "16px 24px", borderTop: `1px solid ${COLORS.border}`, fontSize: 11, color: COLORS.textDim }}>
        Credit Risk PD/LGD/EAD Modeling Platform | Basel III Regulatory Framework | Jay Guwalani
      </div>
    </div>
  );
}
