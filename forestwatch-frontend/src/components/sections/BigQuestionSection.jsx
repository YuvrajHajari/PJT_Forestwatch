import { useState } from "react";
import { OUR_PIPELINE, PAPERS } from "../../data/papers";
import { INDIA_VIEWBOX, INDIA_OUTLINE, CITIES } from "../../data/indiaMap";
import FormulaCompareNote from "../FormulaCompareNote";

function PipelineCard({ p, highlight }) {
  return (
    <div className={`paper-card${highlight ? " ours" : ""}`}>
      <div className="paper-card-name">{p.name}</div>
      <div className="paper-card-row"><span>Area</span><span>{p.area}</span></div>
      <div className="paper-card-row"><span>Method</span><span>{p.method}</span></div>
      <div className="paper-card-row"><span>Horizon</span><span>{p.horizon}</span></div>
      <div className="paper-card-kappa">
        <span className="paper-card-kappa-value">{p.kappa.toFixed(3)}</span>
        <span className="paper-card-kappa-label">Kappa</span>
      </div>
      {p.note && <div className="paper-card-note">{p.note}</div>}
    </div>
  );
}

function KappaBarChart({ selected }) {
  const all = [OUR_PIPELINE, ...PAPERS];
  const w = 760;
  const h = 300;
  const padL = 40, padB = 70, padT = 20;
  const plotH = h - padB - padT;
  const barW = (w - padL) / all.length - 14;
  const yFor = (v) => padT + plotH * (1 - v);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="bq-chart">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={padL} x2={w} y1={yFor(g)} y2={yFor(g)} className="chart-grid" />
          <text x={padL - 8} y={yFor(g) + 3} className="chart-axis-label" textAnchor="end">{g.toFixed(2)}</text>
        </g>
      ))}
      {all.map((p, i) => {
        const x = padL + i * (barW + 14) + 7;
        const isOurs = p === OUR_PIPELINE;
        const isSel = selected && selected.name === p.name;
        return (
          <g key={p.name}>
            <rect
              x={x} y={yFor(p.kappa)} width={barW} height={plotH - (yFor(p.kappa) - padT)}
              className={`chart-bar${isOurs ? " chart-bar-caann" : " chart-bar-baseline"}${isSel ? " chart-bar-selected" : ""}`}
            />
            <text x={x + barW / 2} y={yFor(p.kappa) - 6} className="chart-bar-label" textAnchor="middle">
              {p.kappa.toFixed(2)}
            </text>
            <text x={x + barW / 2} y={h - padB + 18} className="chart-axis-label small" textAnchor="middle">
              {isOurs ? "Ours" : p.name.split(" ")[0]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function BigQuestionSection() {
  const [selectedPaper, setSelectedPaper] = useState(null);
  const shown = selectedPaper || OUR_PIPELINE;

  return (
    <section className="feature-section bq-section">
      <div className="section-eyebrow">For the examiners</div>
      <h2 className="bq-headline">
        Checking old images and predicting the future. <em>What's actually new here?</em>
      </h2>

      <div className="bq-intro-row">
        <svg className="bq-mini-map" viewBox={INDIA_VIEWBOX}>
          <path d={INDIA_OUTLINE} className="map-outline" />
          {CITIES.map((c) => (
            <circle key={c.key} cx={c.x} cy={c.y} r="12" className="map-pin-dot" />
          ))}
        </svg>
        <p className="bq-intro-text">
          Four Indian cities, ten years of real Sentinel-2 reflectance each, a spectral formula
          we designed ourselves, and a prediction model that separates two questions most
          cellular-automata papers fold into one: <strong>how much</strong> will change (a
          historical rate) and <strong>which pixels</strong> will change (a trained network,
          optionally sharpened with a geospatial foundation model). That separation, plus
          testing a foundation model honestly rather than assuming it helps, is the novel part.
        </p>
      </div>

      <div className="bq-formula-recap">
        <div className="formula-card">
          <div className="formula-card-label">Our formula</div>
          <div className="formula-card-eq">FVI = (NIR+G&minus;R&minus;B) / (NIR+G+R+B)</div>
        </div>
        <div className="formula-card">
          <div className="formula-card-label">Our architecture</div>
          <div className="formula-card-eq">Markov rate &times; CA-ANN ranking</div>
        </div>
        <div className="formula-card">
          <div className="formula-card-label">Our extension</div>
          <div className="formula-card-eq">+ Clay v1.5, tested, not assumed</div>
        </div>
      </div>

      <FormulaCompareNote />

      <div className="bq-compare-layout">
        <aside className="bq-sidebar">
          <div className="bq-sidebar-label">Compare against</div>
          <button
            className={`bq-sidebar-item${!selectedPaper ? " active" : ""}`}
            onClick={() => setSelectedPaper(null)}
          >
            This work
          </button>
          {PAPERS.map((p) => (
            <button
              key={p.name}
              className={`bq-sidebar-item${selectedPaper?.name === p.name ? " active" : ""}`}
              onClick={() => setSelectedPaper(p)}
            >
              {p.name}
            </button>
          ))}
        </aside>
        <div className="bq-card-slot">
          <PipelineCard p={shown} highlight={!selectedPaper} />
        </div>
      </div>

      <div className="section-eyebrow">Validated Kappa, one-year horizon</div>
      <KappaBarChart selected={selectedPaper} />
      <p className="bq-closing">
        This pipeline reaches a one-year Kappa of <strong>{OUR_PIPELINE.kappa.toFixed(3)}</strong>,
        ahead of four of the six published configurations compared here, while being the only
        one validated across four cities, checked against an independent external dataset
        (Hansen Global Forest Change), and deployed as a system you can query live rather than a
        one-off offline script.
      </p>
    </section>
  );
}
