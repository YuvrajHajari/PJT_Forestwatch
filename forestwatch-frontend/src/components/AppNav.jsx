import { useEffect, useRef, useState } from "react";

const MORE_ITEMS = [
  { key: "risk", label: "Risk heatmap" },
  { key: "conservation", label: "Conservation priority" },
  { key: "citycompare", label: "City vs city" },
];

export default function AppNav({ cityLabel, section, onNavigate, onChangeCity }) {
  const [moreOpen, setMoreOpen] = useState(false);
  const wrapRef = useRef(null);
  const inMore = ["risk", "conservation", "citycompare"].includes(section);

  useEffect(() => {
    if (!moreOpen) return;
    const onDocClick = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setMoreOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [moreOpen]);

  return (
    <header className="app-nav">
      <div className="app-nav-brand" onClick={onChangeCity} role="button" tabIndex={0}>
        Forest<em>Watch</em> <span className="app-nav-city">{cityLabel}</span>
      </div>
      <nav className="app-nav-links">
        <button
          className={`nav-link${section === "single" ? " active" : ""}`}
          onClick={() => onNavigate("single")}
        >
          Detection
        </button>
        <button
          className={`nav-link${section === "compare" ? " active" : ""}`}
          onClick={() => onNavigate("compare")}
        >
          Compare years
        </button>
        <button
          className={`nav-link${section === "predict" ? " active" : ""}`}
          onClick={() => onNavigate("predict")}
        >
          Prediction
        </button>
        <div className="nav-dropdown" ref={wrapRef}>
          <button
            className={`nav-link${inMore ? " active" : ""}`}
            onClick={() => setMoreOpen((o) => !o)}
          >
            More features {moreOpen ? "▴" : "▾"}
          </button>
          {moreOpen && (
            <div className="nav-dropdown-menu">
              {MORE_ITEMS.map((m) => (
                <button
                  key={m.key}
                  className={`nav-dropdown-item${section === m.key ? " active" : ""}`}
                  onClick={() => {
                    onNavigate(m.key);
                    setMoreOpen(false);
                  }}
                >
                  {m.label}
                </button>
              ))}
            </div>
          )}
        </div>
        <button
          className={`nav-link nav-link-accent${section === "bigquestion" ? " active" : ""}`}
          onClick={() => onNavigate("bigquestion")}
        >
          The big question
        </button>
      </nav>
    </header>
  );
}
