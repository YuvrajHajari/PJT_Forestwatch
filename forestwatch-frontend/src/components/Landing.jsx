import { useState } from "react";
import { INDIA_VIEWBOX, INDIA_OUTLINE, CITIES } from "../data/indiaMap";

export default function Landing({ onSelectCity }) {
  const [hovered, setHovered] = useState(null);
  const [picked, setPicked] = useState(null);

  const handlePick = (key) => {
    setPicked(key);
    // Let the zoom animation play before the parent swaps views.
    setTimeout(() => onSelectCity(key), 650);
  };

  const pickedCity = CITIES.find((c) => c.key === picked);
  const [, , viewW, viewH] = INDIA_VIEWBOX.split(" ").map(Number);

  return (
    <div className="landing">
      <div className="landing-forest-side left" aria-hidden="true" />
      <div className="landing-forest-side right" aria-hidden="true" />

      <div className="landing-hero">
        <div className="section-eyebrow">Sentinel-2 &middot; Computer vision &middot; CA-ANN</div>
        <h1 className="landing-title">
          Forest<em>Watch</em> India
        </h1>
        <p className="landing-tagline">
          Ten years of satellite imagery. One custom formula, one trained model.
          A city&rsquo;s forest, measured and projected forward.
        </p>

        <div className="domain-badge">
          <div className="domain-symbol-row">
            <div className="domain-symbol">
              <span className="domain-symbol-glyph">CV</span>
              <span className="domain-symbol-label">Computer Vision</span>
            </div>
            <span className="domain-symbol-plus">+</span>
            <div className="domain-symbol">
              <span className="domain-symbol-glyph">AI</span>
              <span className="domain-symbol-label">Artificial Intelligence</span>
            </div>
          </div>
          <p className="domain-badge-note">
            ForestWatch combines CV and AI, a pairing 5 other published studies in this
            space also use. The difference: ForestWatch is the only one of them that brings a
            geospatial foundation model, Clay v1.5, into that combination.
          </p>
        </div>

        <p className="landing-cta">Pick a city on the map to begin.</p>
      </div>

      <div className={`landing-map-wrap${picked ? " picked" : ""}`}>
        <svg
          className="landing-map"
          viewBox={INDIA_VIEWBOX}
          style={
            picked
              ? {
                  // Percentage-based so it's correct regardless of the
                  // viewBox-to-rendered-size scale factor (raw viewBox
                  // units aren't CSS pixels once the SVG is scaled).
                  transformOrigin: `${(pickedCity.x / viewW) * 100}% ${(pickedCity.y / viewH) * 100}%`,
                }
              : undefined
          }
        >
          <path d={INDIA_OUTLINE} className="map-outline" />
          {CITIES.map((c) => (
            <g
              key={c.key}
              className={`map-pin${c.active ? " map-pin-active" : " map-pin-minor"}${hovered === c.key ? " hovered" : ""}${picked === c.key ? " selected" : ""}`}
              onMouseEnter={() => c.active && setHovered(c.key)}
              onMouseLeave={() => c.active && setHovered(null)}
              onClick={() => c.active && !picked && handlePick(c.key)}
            >
              {c.active && <circle cx={c.x} cy={c.y} r="26" className="map-pin-ring" />}
              <circle cx={c.x} cy={c.y} r={c.active ? 10 : 5} className="map-pin-dot" />
              <text
                x={c.x}
                y={c.active ? c.y + (c.labelBelow ? 46 : -36) : c.y - 13}
                className={`map-pin-label${c.active ? "" : " minor"}`}
                textAnchor="middle"
              >
                {c.label}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="landing-footer-note">
        Four cities validated end to end &middot; 2016&ndash;2025 &middot; 6&nbsp;km study radius each
      </div>
    </div>
  );
}
