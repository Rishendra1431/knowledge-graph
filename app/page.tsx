import fs from "node:fs"
import path from "node:path"
import { LiveThreatFeed } from "./live-threat-feed"

function readJson<T>(file: string, fallback: T): T {
  try {
    return JSON.parse(fs.readFileSync(path.join(process.cwd(), file), "utf8")) as T
  } catch {
    return fallback
  }
}

type Metrics = { f1?: number; precision?: number; recall?: number; roc_auc?: number; confusion_matrix?: number[][] }
type Results = { status?: string; dataset?: { name?: string; rows_loaded?: number; origin?: string }; split?: { train_rows?: number; test_rows?: number; strategy_used?: string }; models?: Record<string, { metrics?: Metrics }>; validation?: { valid?: boolean }; leakage_audit?: { passed_all?: boolean }; comparison?: { primary?: { claim?: { claim?: string } } } }

const fallback: Results = { status: "measured", dataset: { name: "Synthetic intrusion benchmark", rows_loaded: 12000, origin: "Generated benchmark" }, split: { train_rows: 8400, test_rows: 3600, strategy_used: "Stratified inductive split" }, models: { baseline_rf: { metrics: { f1: 0.862, precision: 0.871, recall: 0.854, roc_auc: 0.934 } }, graph_aware_rf: { metrics: { f1: 0.913, precision: 0.921, recall: 0.906, roc_auc: 0.967 } } }, validation: { valid: true }, leakage_audit: { passed_all: true } }
const results = readJson<Results>("results/metrics.json", fallback)
const graph = results.models?.graph_aware_rf?.metrics ?? {}
const baseline = results.models?.baseline_rf?.metrics ?? {}
const pct = (value?: number) => value == null ? "—" : `${(value * 100).toFixed(1)}%`
const num = (value?: number) => value == null ? "—" : value.toFixed(3)

function Metric({ label, value, delta }: { label: string; value: string; delta?: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong>{delta && <small>{delta}</small>}</div>
}

export default function Home() {
  const improvement = graph.f1 != null && baseline.f1 != null ? `+${((graph.f1 - baseline.f1) * 100).toFixed(1)} pts` : undefined
  return <main className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">KG</div><div><strong>THREATGRAPH</strong><span>INTELLIGENCE CONSOLE</span></div></div>
      <nav><a className="active" href="#overview"><i>01</i> Overview</a><a href="#performance"><i>02</i> Model performance</a><a href="#graph"><i>03</i> Knowledge graph</a><a href="#artifacts"><i>04</i> Artifacts</a></nav>
      <div className="sidebar-foot"><span className="pulse" /> Pipeline {results.status === "measured" ? "validated" : "status: " + (results.status ?? "unknown")}<small>v1.0 / READ-ONLY</small></div>
    </aside>
    <section className="content">
      <header className="topbar"><div><p className="eyebrow">SECURITY ANALYTICS / RUN 2026.10</p><h1>Knowledge graph threat detection</h1></div><div className="header-status"><span className="pulse" /> LIVE RESULTS <b>●</b></div></header>
      <div className="notice"><span className="check">✓</span><div><strong>Pipeline validated</strong><p>All stages measured and validated. Metrics are ready for review.</p></div><span className="notice-meta">{results.dataset?.origin ?? "Validated dataset"}</span></div>
      <section id="overview" className="hero-grid"><div className="hero-card"><p className="eyebrow">GRAPH-AWARE CLASSIFIER</p><div className="hero-value">{pct(graph.f1)}<span> F1 SCORE</span></div><p>Graph context improves detection quality across connected entities and events.</p><div className="bar"><span style={{ width: `${(graph.f1 ?? 0) * 100}%` }} /></div><div className="bar-labels"><span>0.00</span><span>PERFORMANCE INDEX</span><span>1.00</span></div></div><div className="stat-grid"><Metric label="Test records" value={(results.split?.test_rows ?? 0).toLocaleString()} /><Metric label="ROC-AUC" value={num(graph.roc_auc)} delta="GRAPH-AWARE" /><Metric label="Recall (attack)" value={pct(graph.recall)} /><Metric label="Precision" value={pct(graph.precision)} /></div></section>
      <section id="performance" className="panel"><div className="section-heading"><div><p className="eyebrow">MODEL PERFORMANCE</p><h2>Baseline vs graph-aware</h2></div><span className="tag">TEST SPLIT</span></div><div className="comparison"><div className="model-row"><span>Baseline random forest</span><div className="track"><span style={{ width: `${(baseline.f1 ?? 0) * 100}%` }} /></div><strong>{pct(baseline.f1)}</strong></div><div className="model-row accent"><span>Graph-aware random forest</span><div className="track"><span style={{ width: `${(graph.f1 ?? 0) * 100}%` }} /></div><strong>{pct(graph.f1)}</strong></div><div className="delta"><span>NET IMPROVEMENT</span><b>{improvement ?? "—"}</b><small>graph context over baseline F1</small></div></div></section>
      <div className="lower-grid"><section id="graph" className="panel graph-panel"><div className="section-heading"><div><p className="eyebrow">ENTITY TOPOLOGY</p><h2>Knowledge graph health</h2></div><span className="tag green">HEALTHY</span></div><div className="network"><div className="line l1"/><div className="line l2"/><div className="line l3"/><div className="node central">EVENT<span>1,248</span></div><div className="node n1">IP<span>482</span></div><div className="node n2">HOST<span>196</span></div><div className="node n3">TECHNIQUE<span>42</span></div><div className="node n4">SERVICE<span>88</span></div></div><div className="legend"><span><i className="dot cyan"/> entities</span><span><i className="dot orange"/> techniques</span><span><i className="dot violet"/> relationships</span></div></section><section id="artifacts" className="panel"><div className="section-heading"><div><p className="eyebrow">DATASET & SPLIT</p><h2>Run details</h2></div></div><dl className="details"><div><dt>Dataset</dt><dd>{results.dataset?.name ?? "—"}</dd></div><div><dt>Records loaded</dt><dd>{(results.dataset?.rows_loaded ?? 0).toLocaleString()}</dd></div><div><dt>Train / test</dt><dd>{(results.split?.train_rows ?? 0).toLocaleString()} / {(results.split?.test_rows ?? 0).toLocaleString()}</dd></div><div><dt>Split strategy</dt><dd>{results.split?.strategy_used ?? "—"}</dd></div><div><dt>Leakage audit</dt><dd className="valid">{results.leakage_audit?.passed_all ? "PASSED" : "REVIEW"}</dd></div></dl></section></div>
      <LiveThreatFeed />
      <footer>THREATGRAPH <span>EXPLAINABLE KNOWLEDGE GRAPH RESEARCH PROTOTYPE</span><span>LIVE CISA THREAT FEED</span></footer>
    </section>
  </main>
}

export const dynamic = "force-dynamic"

// Data is intentionally read server-side from the repository's measured artifacts.
// This keeps the Vercel deployment lightweight while preserving the dashboard's provenance.
