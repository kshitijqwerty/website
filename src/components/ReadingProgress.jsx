import { useEffect, useRef } from "react";
import { useTheme } from "../context/ThemeContext";

export default function ReadingProgress({ articleRef }) {
  const { resolved: theme } = useTheme();
  const barRef = useRef(null);
  const isLight = theme === "light";

  useEffect(() => {
    const el = articleRef.current;
    if (!el || !barRef.current) return;

    function update() {
      if (!barRef.current) return;
      const { top, height } = el.getBoundingClientRect();
      const winH = window.innerHeight;
      const total = height - winH;
      const scrolled = -top;
      const pct = total > 0 ? Math.min(1, Math.max(0, scrolled / total)) : 0;
      barRef.current.style.width = `${pct * 100}%`;
    }

    let ticking = false;
    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        update();
        ticking = false;
      });
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    update();

    return () => window.removeEventListener("scroll", onScroll);
  }, [articleRef]);

  return (
    <div className="fixed top-0 left-0 right-0 z-[60] h-[3px] pointer-events-none">
      <div
        ref={barRef}
        className="h-full"
        style={{
          backgroundColor: isLight ? "#404040" : "#e5e5e5",
          boxShadow: isLight
            ? "0 0 6px 2px rgba(0,0,0,0.15)"
            : "0 0 6px 2px rgba(255,255,255,0.25)",
        }}
      />
    </div>
  );
}
