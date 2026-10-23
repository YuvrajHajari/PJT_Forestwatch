import { useEffect, useState } from "react";
import { api, bust } from "../../api";
import ProcessLog from "../ProcessLog";
import FragmentationGrid from "../FragmentationGrid";
import PipelineStrip from "../PipelineStrip";
import { stepsFor } from "../../processSteps";

export default function SingleYearSection({ city, cityLabel, years }) {
  const [year, setYear] = useState(years[years.length - 1]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState("overlay");

  useEffect(() => {
    if (!year) return;
    setLoading(true);
    api
      .analyze(city, year)
      .then(setData)
      .finally(() => setLoading(false));
  }, [city, year]);

  return (
    <section className="feature-section">
      <div className="section-eyebrow">Single-year detection</div>
      <h2 className="section-title">{cityLabel}, pick any fetched year</h2>
      <PipelineStrip mode="single" />

      <div className="year-strip">
        {years.map((y) => (
          <button key={y} className={`year-chip${year === y ? " active" : ""}`} onClick={() => setYear(y)}>
            {y}
          </button>
        ))}
      </div>

      <div className="two-col">
        <div className="mask-viewer">
          {loading && (
            <div className="stage-loader embedded">
              <ProcessLog steps={stepsFor("single", { city: cityLabel, year })} />
            </div>
          )}
          {data && !loading && (
            <>
              <div className="view-strip static">
                {["satellite", "overlay", "mask"].map((v) => (
                  <button key={v} className={`vbtn${view === v ? " active" : ""}`} onClick={() => setView(v)}>
                    {v}
                  </button>
                ))}
              </div>
              <img className="mask-viewer-img" src={bust(data[`${view}_url`])} alt={view} />
              <div className="img-label left">{cityLabel.toUpperCase()} &middot; {year}</div>
            </>
          )}
        </div>

        {data && !loading && (
          <div className="side-stats">
            <div className="big-stat">
              <div className="big-stat-value">{data.forest_area_ha}</div>
              <div className="big-stat-unit">hectares of forest</div>
              <div className="big-stat-label">{data.coverage_pct}% of the study area</div>
            </div>
            <div className="metric-grid">
              <div className="metric-cell">
                <div className="metric-cell-label">Carbon stock</div>
                <div className="metric-cell-value">{data.carbon_t.toLocaleString("en-IN")}</div>
                <div className="metric-cell-unit">t-C</div>
              </div>
              <div className="metric-cell">
                <div className="metric-cell-label">CO&#8322; equivalent</div>
                <div className="metric-cell-value">{data.co2_equiv_t.toLocaleString("en-IN")}</div>
                <div className="metric-cell-unit">t-CO&#8322;e</div>
              </div>
            </div>
            <FragmentationGrid fragmentation={data.fragmentation} />
          </div>
        )}
      </div>
    </section>
  );
}
