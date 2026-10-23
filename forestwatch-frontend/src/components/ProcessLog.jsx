import { useEffect, useRef, useState } from "react";

// Reveals `steps` one at a time while the real request behind it is in
// flight, instead of a generic spinner. Timing is cosmetic (a single HTTP
// call doesn't expose per-stage progress), but the content isn't: every
// line names a real stage of the pipeline the backend is actually running,
// in its actual order (see processSteps.js). The log holds on the final
// step until the response actually lands, rather than racing ahead of it.
export default function ProcessLog({ steps }) {
  const [visible, setVisible] = useState(1);
  const [heldFor, setHeldFor] = useState(0); // seconds spent on the final, held step
  const timer = useRef(null);

  useEffect(() => {
    setVisible(1);
    setHeldFor(0);
    let i = 1;
    const tick = () => {
      if (i >= steps.length - 1) return; // hold on the last line
      i += 1;
      setVisible(i);
      const next = 320 + Math.random() * 260;
      timer.current = setTimeout(tick, next);
    };
    timer.current = setTimeout(tick, 320 + Math.random() * 260);
    return () => clearTimeout(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [steps.join("|")]);

  // Some pipelines genuinely take tens of seconds (the CA-ANN ensemble
  // scoring stage, in particular) -- once the log reaches that final step
  // there's no further stage to reveal, so a real elapsed-time readout
  // replaces a progress bar we have no honest way to fill in.
  useEffect(() => {
    if (visible < steps.length - 1) return;
    const start = Date.now();
    const id = setInterval(() => setHeldFor(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [visible, steps.length]);

  return (
    <div className="process-log">
      {steps.slice(0, visible).map((s, i) => {
        const isLast = i === visible - 1;
        const showElapsed = isLast && visible >= steps.length - 1 && heldFor >= 3;
        return (
          <div className={`process-line${isLast ? " active" : ""}`} key={i}>
            <span className="process-marker">{isLast ? "›" : "✓"}</span>
            <span>
              {s}
              {showElapsed && <span className="process-elapsed"> · {heldFor}s</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
}
