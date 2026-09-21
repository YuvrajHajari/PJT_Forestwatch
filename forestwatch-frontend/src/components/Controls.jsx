export default function Controls({
  modes,
  mode,
  setMode,
  cities,
  city,
  setCity,
  compareCityB,
  setCompareCityB,
  year,
  setYear,
  yearA,
  setYearA,
  yearB,
  setYearB,
  targetYear,
  setTargetYear,
  hansenYear,
  setHansenYear,
  years,
  targetYears,
  onRun,
  loading,
  error,
}) {
  return (
    <div className="controls">
      <div className="ctrl-group">
        <span className="ctrl-label">City</span>
        <select value={city} onChange={(e) => setCity(e.target.value)}>
          {cities.map((c) => (
            <option key={c.key} value={c.key}>
              {c.label}
            </option>
          ))}
        </select>
      </div>

      <div className="ctrl-group">
        <span className="ctrl-label">Mode</span>
        <div className="mode-tabs">
          {modes.map((m) => (
            <button
              key={m.key}
              className={`mode-tab${mode === m.key ? " active" : ""}`}
              onClick={() => setMode(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {mode === "citycompare" && (
        <div className="ctrl-group">
          <span className="ctrl-label">Compare against</span>
          <select value={compareCityB} onChange={(e) => setCompareCityB(e.target.value)}>
            {cities
              .filter((c) => c.key !== city)
              .map((c) => (
                <option key={c.key} value={c.key}>
                  {c.label}
                </option>
              ))}
          </select>
        </div>
      )}

      {mode === "single" && (
        <div className="ctrl-group">
          <span className="ctrl-label">Year</span>
          <select value={year} onChange={(e) => setYear(e.target.value)}>
            {years.map((y) => (
              <option key={y}>{y}</option>
            ))}
          </select>
        </div>
      )}

      {mode === "compare" && (
        <div className="ctrl-group">
          <span className="ctrl-label">From</span>
          <select value={yearA} onChange={(e) => setYearA(e.target.value)}>
            {years.map((y) => (
              <option key={y}>{y}</option>
            ))}
          </select>
          <span className="ctrl-label" style={{ marginLeft: 4 }}>
            To
          </span>
          <select value={yearB} onChange={(e) => setYearB(e.target.value)}>
            {years.map((y) => (
              <option key={y}>{y}</option>
            ))}
          </select>
        </div>
      )}

      {mode === "predict" && (
        <div className="ctrl-group">
          <span className="ctrl-label">Target year</span>
          <select value={targetYear} onChange={(e) => setTargetYear(e.target.value)}>
            {targetYears.map((y) => (
              <option key={y}>{y}</option>
            ))}
          </select>
        </div>
      )}

      {mode === "hansen" && (
        <div className="ctrl-group">
          <span className="ctrl-label">Year</span>
          <select value={hansenYear} onChange={(e) => setHansenYear(e.target.value)}>
            {years.map((y) => (
              <option key={y}>{y}</option>
            ))}
          </select>
        </div>
      )}

      <button className="run-btn" onClick={onRun} disabled={loading}>
        {loading ? "Running…" : "Analyse →"}
      </button>
      {error && <span className="status-msg error">Error: {error}</span>}
    </div>
  );
}
