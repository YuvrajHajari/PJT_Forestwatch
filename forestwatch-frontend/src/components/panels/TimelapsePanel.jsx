import { cap } from "../../narrative";

export default function TimelapsePanel({ data }) {
  const { city, years } = data;
  return (
    <div className="panel-section">
      <div className="section-eyebrow">Time-lapse</div>
      <p className="narrative">
        Drag the slider or press play to step through {years.length} years of raw Sentinel-2
        imagery ({years[0]}–{years[years.length - 1]}) for {cap(city)}, showing urban growth and
        vegetation change directly in the satellite composite.
      </p>
    </div>
  );
}
