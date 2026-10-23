// Pre-rendered diagrams (built and exported separately, see
// scratchpad/diagrams.html in the project history), not generated at
// runtime. This component just picks the right static image.
const IMAGES = [
  "/diagrams/bts-1-clay.png",
  "/diagrams/bts-2-handbuilt.png",
  "/diagrams/bts-3-merge.png",
  "/diagrams/bts-4-network.png",
  "/diagrams/bts-5-history.png",
  "/diagrams/bts-6-combine.png",
];

export default function BtsStepDiagram({ step }) {
  return <img className="bts-diagram" src={IMAGES[step] || IMAGES[0]} alt="" />;
}
