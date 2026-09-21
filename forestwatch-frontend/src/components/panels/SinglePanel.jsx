import { useEffect, useRef } from "react";
import FragmentationGrid from "../FragmentationGrid";
import { singleYearNarrative } from "../../narrative";

function TrendBars({ trendData }) {
  const years = Object.keys(trendData).sort();
  const barsRef = useRef(null);

  useEffect(() => {
    if (!barsRef.current) return;
    const fills = barsRef.current.querySelectorAll(".bar-fill");
    // Animate width in on the next tick, same as the original vanilla-JS
    // implementation -- keeps the single load-in flourish, nothing else.
    requestAnimationFrame(() => {
      fills.forEach((f) => {
        f.style.width = f.dataset.w;
      });
    });
  }, [trendData]);

  if (years.length === 0) return null;
  const max = Math.max(...years.map((y) => trendData[y]));

  return (
    <div className="compare-bars" ref={barsRef}>
      {years.map((y) => (
        <div className="bar-row" key={y}>
          <span className="bar-year">{y}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              data-w={`${Math.round((trendData[y] / max) * 100)}%`}
              style={{ width: 0 }}
            />
          </div>
          <span className="bar-val">{trendData[y]} ha</span>
        </div>
      ))}
    </div>
  );
}

export default function SinglePanel({ data, trendData }) {
  return (
    <>
      <div className="panel-section">
        <div className="section-eyebrow">Summary</div>
        <p className="narrative">{singleYearNarrative(data, trendData)}</p>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Forest cover</div>
        <div className="big-stat">
          <div className="big-stat-value">{data.forest_area_ha.toLocaleString("en-IN")}</div>
          <div className="big-stat-unit">hectares</div>
          <div className="big-stat-label">Vegetated land detected within study area</div>
        </div>
        <div className="metric-grid">
          <div className="metric-cell">
            <div className="metric-cell-label">Coverage</div>
            <div className="metric-cell-value">{data.coverage_pct}</div>
            <div className="metric-cell-unit">% of frame</div>
          </div>
          <div className="metric-cell">
            <div className="metric-cell-label">Carbon stock</div>
            <div className="metric-cell-value">{data.carbon_t.toLocaleString("en-IN")}</div>
            <div className="metric-cell-unit">t-C (IPCC)</div>
          </div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">CO₂ equivalent</div>
        <div className="big-stat">
          <div className="big-stat-value">{data.co2_equiv_t.toLocaleString("en-IN")}</div>
          <div className="big-stat-unit">tonnes-CO₂e stored</div>
          <div className="big-stat-label">Carbon × 3.67 molecular weight ratio</div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Landscape fragmentation</div>
        <FragmentationGrid fragmentation={data.fragmentation} />
      </div>

      <div className="panel-section">
        <div className="section-eyebrow">Historical trend</div>
        <TrendBars trendData={trendData} />
      </div>
    </>
  );
}
