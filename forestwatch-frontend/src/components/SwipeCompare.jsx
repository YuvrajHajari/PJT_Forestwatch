import { useRef, useState, useEffect, useCallback } from "react";

export default function SwipeCompare({ imgA, imgB, labelA, labelB }) {
  const wrapRef = useRef(null);
  const [pct, setPct] = useState(50);
  const [dragging, setDragging] = useState(false);

  const updateFromClientX = useCallback((clientX) => {
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
    setPct((x / rect.width) * 100);
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e) => {
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      updateFromClientX(clientX);
    };
    const onUp = () => setDragging(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("touchmove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchend", onUp);
    };
  }, [dragging, updateFromClientX]);

  const startDrag = (e) => {
    setDragging(true);
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    updateFromClientX(clientX);
  };

  return (
    <div className="swipe-wrap" ref={wrapRef} onMouseDown={startDrag} onTouchStart={startDrag}>
      <img className="swipe-img" src={imgB} alt={labelB} draggable={false} />
      <img
        className="swipe-img"
        src={imgA}
        alt={labelA}
        draggable={false}
        style={{ clipPath: `inset(0 ${100 - pct}% 0 0)` }}
      />
      <div className="swipe-handle" style={{ left: `${pct}%` }} />
      <span className="img-label left">{labelA}</span>
      <span className="img-label right">{labelB}</span>
    </div>
  );
}
