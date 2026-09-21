import SinglePanel from "./panels/SinglePanel";
import ComparePanel from "./panels/ComparePanel";
import PredictPanel from "./panels/PredictPanel";
import RiskPanel from "./panels/RiskPanel";
import ConservationPanel from "./panels/ConservationPanel";
import HansenPanel from "./panels/HansenPanel";
import TimelapsePanel from "./panels/TimelapsePanel";
import CityComparePanel from "./panels/CityComparePanel";

const PANELS = {
  single: SinglePanel,
  compare: ComparePanel,
  predict: PredictPanel,
  risk: RiskPanel,
  conservation: ConservationPanel,
  hansen: HansenPanel,
  timelapse: TimelapsePanel,
  citycompare: CityComparePanel,
};

export default function SidePanel({ mode, data, trendData }) {
  const Panel = PANELS[mode];

  if (!data) {
    return (
      <aside className="panel">
        <div className="empty-state">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="14" stroke="#bbb" strokeWidth="1.5" />
            <path d="M16 10v6l4 4" stroke="#bbb" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          Run an analysis to see results
        </div>
      </aside>
    );
  }

  return (
    <aside className="panel">
      <Panel data={data} trendData={trendData} />
    </aside>
  );
}
