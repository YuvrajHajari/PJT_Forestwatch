import { useEffect, useState } from "react";
import { api, bust } from "../../api";
import ProcessLog from "../ProcessLog";
import FormulaCompareNote from "../FormulaCompareNote";

const CV_STEPS = [
  "Loading the real surface-reflectance bands (NIR, Green, Red, Blue)",
  "Computing FVI = (NIR+G−R−B) / (NIR+G+R+B) for every pixel",
  "Running three-class Otsu thresholding on the FVI histogram",
  "Cleaning the mask with morphological open/close (OpenCV)",
];

export default function MaskExplainer({ city, cityLabel, year, onContinue }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState("satellite");

  useEffect(() => {
    if (!city || !year) return;
    setLoading(true);
    setData(null);
    api
      .analyze(city, year)
      .then(setData)
      .finally(() => setLoading(false));
  }, [city, year]);

  return (
    <section className="intro-step">
      <div className="section-eyebrow">Step 2 &middot; Vegetation detection</div>
      <h2 className="intro-title">Turning a photo into a forest measurement</h2>

      <div className="cv-badge">
        Computer vision only, here &middot; OpenCV + a spectral formula &middot; no neural
        network, no AI
      </div>

      <div className="formula-row">
        <div className="formula-card">
          <div className="formula-card-label">The formula</div>
          <div className="formula-card-eq">FVI = (NIR+G&minus;R&minus;B) / (NIR+G+R+B)</div>
          <div className="formula-card-note">
            Near-infrared folded directly in, so water (absorbs NIR) and vegetation (reflects
            NIR) separate in one pass, no second water-mask formula needed.
          </div>
        </div>
        <div className="formula-card">
          <div className="formula-card-label">The split</div>
          <div className="formula-card-eq">3-class Otsu threshold</div>
          <div className="formula-card-note">
            Finds two cut points in the FVI histogram automatically, water, built-up land, and
            forest, recomputed per image rather than hand-tuned.
          </div>
        </div>
      </div>

      <FormulaCompareNote />

      {loading && (
        <div className="intro-process-wrap">
          <ProcessLog steps={CV_STEPS} />
        </div>
      )}

      {data && !loading && (
        <>
          <div className="mask-viewer">
            <div className="view-strip static">
              {["satellite", "overlay", "mask"].map((v) => (
                <button
                  key={v}
                  className={`vbtn${view === v ? " active" : ""}`}
                  onClick={() => setView(v)}
                >
                  {v}
                </button>
              ))}
            </div>
            <img className="mask-viewer-img" src={bust(data[`${view}_url`])} alt={view} />
            <div className="img-label left">{cityLabel.toUpperCase()} &middot; {year}</div>
          </div>

          <div className="metric-grid intro-metrics">
            <div className="metric-cell">
              <div className="metric-cell-label">Forest detected</div>
              <div className="metric-cell-value">{data.forest_area_ha}</div>
              <div className="metric-cell-unit">hectares</div>
            </div>
            <div className="metric-cell">
              <div className="metric-cell-label">Coverage</div>
              <div className="metric-cell-value">{data.coverage_pct}%</div>
              <div className="metric-cell-unit">of study area</div>
            </div>
            <div className="metric-cell">
              <div className="metric-cell-label">Carbon stock</div>
              <div className="metric-cell-value">{data.carbon_t.toLocaleString("en-IN")}</div>
              <div className="metric-cell-unit">t-C</div>
            </div>
            <div className="metric-cell">
              <div className="metric-cell-label">Patches</div>
              <div className="metric-cell-value">{data.fragmentation.patch_count}</div>
              <div className="metric-cell-unit">distinct forest patches</div>
            </div>
          </div>

          <button className="run-btn" onClick={onContinue}>
            Enter ForestWatch
          </button>
        </>
      )}
    </section>
  );
}
