import { useEffect, useRef } from "react";
import { useTheme } from "../context/useTheme";

export default function ReadingProgress() {
  const { resolved: theme } = useTheme();
  const barRef = useRef(null);
  const isLight = theme === "light";

  useEffect(() => {
    const bar = barRef.current;
    if (!bar) return;

    function update() {
      const el = document.querySelector(".article-content");
      if (!el) return;
      const { top, height } = el.getBoundingClientRect();
      const winH = window.innerHeight;
      const total = height - winH;
      const scrolled = -top;
      const pct = total > 0 ? Math.min(1, Math.max(0, scrolled / total)) : 0;
      bar.style.width = `${pct * 100}%`;
    }

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update, { passive: true });

    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);

  return (
    <div className="fixed top-0 left-0 right-0 z-[70] h-[3px] pointer-events-none">
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
