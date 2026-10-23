import { useEffect, useRef, useState } from "react";
import BtsStepDiagram from "./BtsStepDiagrams";

const BEATS = [
  {
    tag: "1",
    title: "Clay v1.5",
    body: "A foundation model, pretrained on millions of satellite images worldwide. We hand it a tile and it hands back 1024 numbers that summarise what's in that image.",
    chip: "frozen · pretrained · 8 numbers after reduction",
  },
  {
    tag: "2",
    title: "Our own 4 numbers",
    body: "Measured directly, no learning involved: how much forest surrounds this pixel at three different distances, and how close it sits to a forest edge.",
    chip: "hand-built · 4 numbers",
  },
  {
    tag: "3",
    title: "12 numbers per pixel",
    body: "Clay's 8 and our 4, concatenated. Every pixel in the image now carries this same 12-number description.",
    chip: "4 + 8 = 12",
  },
  {
    tag: "4",
    title: "Our neural network",
    body: "A small network we trained ourselves on this project's own data, 12 numbers in, one score out: how likely is this specific pixel to change? This is the only part that learns.",
    chip: "trained by us · 12 → 16 → 8 → 1",
  },
  {
    tag: "5",
    title: "10 years of history",
    body: "Separately, we look at how much forest typically changed year to year across the whole decade. This sets a budget: how many pixels should change, not which ones.",
    chip: "no learning · simple counting",
  },
  {
    tag: "6",
    title: "Put together",
    body: "The network ranks every pixel by risk. The history sets how many pixels get to change. Rank the candidates, spend the budget on the riskiest ones first, that's the projected map.",
    chip: "network says WHERE · history says HOW MUCH",
  },
];

// Scroll-driven: each beat is its own block in the left column, and a
// thin trigger band at viewport centre (via IntersectionObserver
// rootMargin) decides which one is "active" as the user scrolls, no
// click required. The right column stays sticky and swaps in a small
// diagram for whichever step is active.
export default function ClayFlowDiagram() {
  const [active, setActive] = useState(0);
  const refs = useRef([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = refs.current.indexOf(entry.target);
            if (idx !== -1) setActive(idx);
          }
        });
      },
      { rootMargin: "-40% 0px -40% 0px", threshold: 0 }
    );
    refs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="bts-scrolly">
      <div className="bts-scrolly-steps">
        {BEATS.map((b, i) => (
          <div
            key={b.tag}
            ref={(el) => {
              refs.current[i] = el;
            }}
            className={`bts-step${active === i ? " active" : ""}`}
          >
            <div className="flow-node-tag">{b.tag}</div>
            <div className="flow-node-title">{b.title}</div>
            <div className="flow-node-chip">{b.chip}</div>
            <div className="flow-node-body">{b.body}</div>
          </div>
        ))}
      </div>
      <div className="bts-scrolly-sticky">
        <div className="bts-diagram-frame">
          <BtsStepDiagram step={active} />
        </div>
        <div className="bts-diagram-caption">
          Step {active + 1} of {BEATS.length} &middot; {BEATS[active].title}
        </div>
      </div>
    </div>
  );
}

export function BtsSummaryDiagram() {
  return (
    <div className="bts-summary">
      <img className="bts-summary-img" src="/diagrams/bts-7-summary.png" alt="" />
    </div>
  );
}
