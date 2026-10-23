import { useEffect, useRef, useState } from "react";
import { satelliteUrl, bust } from "../../api";

// Step 1 of the per-city intro: narrates the actual fetch source one year
// at a time, then opens into a switchable collage. No fake data -- every
// thumbnail is the real cached satellite_{city}_{year}.png the backend
// already serves, revealed on a timer rather than all at once.
export default function FetchImages({ city, cityLabel, years, onContinue }) {
  const [started, setStarted] = useState(false);
  const [revealed, setRevealed] = useState(0);
  const [active, setActive] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    setStarted(false);
    setRevealed(0);
    setActive(null);
  }, [city]);

  const start = () => {
    setStarted(true);
    let i = 0;
    const tick = () => {
      i += 1;
      setRevealed(i);
      if (i < years.length) {
        timer.current = setTimeout(tick, 420);
      } else {
        setActive(years[years.length - 1]);
      }
    };
    timer.current = setTimeout(tick, 420);
  };

  useEffect(() => () => clearTimeout(timer.current), []);

  const allRevealed = revealed >= years.length;

  return (
    <section className="intro-step">
      <div className="section-eyebrow">Step 1 &middot; Data acquisition</div>
      <h2 className="intro-title">Fetching ten years of {cityLabel}</h2>
      <p className="intro-body">
        Every image below is real Sentinel-2 surface reflectance, pulled from the Copernicus
        archive through Google Earth Engine, not a stock photo and not a synthetic render. One
        composite per year, 2016 through 2025, each built from a January&ndash;May window to
        keep cloud cover and monsoon greenness out of the comparison.
      </p>

      {!started && (
        <button className="run-btn fetch-btn" onClick={start}>
          Fetch images
        </button>
      )}

      {started && (
        <>
          <div className="fetch-log">
            {years.slice(0, revealed).map((y, i) => (
              <div className={`fetch-log-line${i === revealed - 1 && !allRevealed ? " active" : ""}`} key={y}>
                <span className="process-marker">{i === revealed - 1 && !allRevealed ? "›" : "✓"}</span>
                <span>
                  {y} &middot; Sentinel-2 L2A surface reflectance, Google Earth Engine &middot;
                  1024&times;1024px
                </span>
              </div>
            ))}
          </div>

          <div className="fetch-grid">
            {years.map((y, i) => {
              const shown = i < revealed;
              return (
                <button
                  key={y}
                  className={`fetch-thumb${shown ? " shown" : ""}${active === y ? " active" : ""}`}
                  onClick={() => shown && setActive(y)}
                  disabled={!shown}
                >
                  {shown ? (
                    <img src={satelliteUrl(city, y)} alt={`${cityLabel} ${y}`} />
                  ) : (
                    <span className="fetch-thumb-placeholder" />
                  )}
                  <span className="fetch-thumb-year">{y}</span>
                </button>
              );
            })}
          </div>

          {active && (
            <div className="fetch-focus">
              <div className="year-strip">
                {years.map((y) => (
                  <button
                    key={y}
                    className={`year-chip${active === y ? " active" : ""}`}
                    onClick={() => setActive(y)}
                  >
                    {y}
                  </button>
                ))}
              </div>
              <img
                className="fetch-focus-img"
                src={bust(satelliteUrl(city, active))}
                alt={`${cityLabel} ${active}, full resolution`}
              />
              <div className="img-label left">{cityLabel.toUpperCase()} &middot; {active}</div>
            </div>
          )}

          {allRevealed && active && (
            <button className="run-btn" onClick={onContinue}>
              Next: how the mask gets built
            </button>
          )}
        </>
      )}
    </section>
  );
}
