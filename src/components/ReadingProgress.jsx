import { useEffect, useRef, useState } from "react";
import { useTheme } from "../context/ThemeContext";

export default function ReadingProgress({ articleRef }) {
  const [progress, setProgress] = useState(0);
  const { resolved: theme } = useTheme();
  const raf = useRef();

  useEffect(() => {
    function update() {
      const el = articleRef.current;
      if (!el) {
        setProgress(0);
        return;
      }

      const { top, height } = el.getBoundingClientRect();
      const winH = window.innerHeight;
      const total = height - winH;
      const scrolled = -top;

      setProgress(total > 0 ? Math.min(1, Math.max(0, scrolled / total)) : 0);
    }

    function loop() {
      update();
      raf.current = requestAnimationFrame(loop);
    }

    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [articleRef]);

  const isLight = theme === "light";

  return (
    <div className="fixed top-0 left-0 right-0 z-[60] h-[3px] pointer-events-none">
      <div
        className="h-full"
        style={{
          width: `${progress * 100}%`,
          backgroundColor: isLight ? "#404040" : "#e5e5e5",
          boxShadow: isLight
            ? "0 0 6px 2px rgba(0,0,0,0.15)"
            : "0 0 6px 2px rgba(255,255,255,0.25)",
        }}
      />
    </div>
  );
}
