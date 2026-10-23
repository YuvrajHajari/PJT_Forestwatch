// Pre-rendered pipeline diagrams, one static image per feature section
// (built separately, see scratchpad/diagrams.html in the project
// history, not assembled from a data array at runtime).
const IMAGES = {
  single: "/diagrams/pipeline-single.png",
  compare: "/diagrams/pipeline-compare.png",
  predict: "/diagrams/pipeline-predict.png",
  risk: "/diagrams/pipeline-risk.png",
  conservation: "/diagrams/pipeline-conservation.png",
  citycompare: "/diagrams/pipeline-citycompare.png",
};

export default function PipelineStrip({ mode }) {
  const src = IMAGES[mode];
  if (!src) return null;
  return (
    <div className="pipeline-strip">
      <img src={src} alt="" />
    </div>
  );
}
