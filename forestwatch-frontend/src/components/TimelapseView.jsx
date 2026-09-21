import { useState, useEffect, useRef } from "react";
import { satelliteUrl } from "../api";

export default function TimelapseView({ city, years }) {
  const [idx, setIdx] = useState(Math.max(years.length - 1, 0));
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef(null);

  // Preload every year's image up front so dragging the slider feels
  // instant instead of popping in frame by frame.
  useEffect(() => {
    years.forEach((y) => {
      const img = new Image();
      img.src = satelliteUrl(city, y);
    });
  }, [city, years]);

  useEffect(() => {
    if (!playing) {
      clearInterval(intervalRef.current);
      return;
    }
    intervalRef.current = setInterval(() => {
      setIdx((prev) => (prev + 1) % years.length);
    }, 900);
    return () => clearInterval(intervalRef.current);
  }, [playing, years.length]);

  if (years.length === 0) {
    return (
      <div className="stage-placeholder">
        <span>No years of imagery fetched for this city yet.</span>
      </div>
    );
  }

  return (
    <div style={{ height: "100%", position: "relative" }}>
      <img className="stage-img" src={satelliteUrl(city, years[idx])} alt={`${city} ${years[idx]}`} />
      <span className="img-label left">
        {city.toUpperCase()} · {years[idx]}
      </span>
      <div className="timelapse-controls">
        <button
          className="timelapse-play"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? "❚❚" : "▶"}
        </button>
        <input
          type="range"
          className="timelapse-slider"
          min={0}
          max={years.length - 1}
          value={idx}
          onChange={(e) => {
            setPlaying(false);
            setIdx(Number(e.target.value));
          }}
        />
        <span className="timelapse-year">{years[idx]}</span>
      </div>
    </div>
  );
}
