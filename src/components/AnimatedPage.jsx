import { useEffect } from "react";
import { useLocation } from "react-router-dom";

export default function AnimatedPage({ children }) {
  const { hash } = useLocation();

  useEffect(() => {
    if (hash) {
      const id = hash.replace("#", "");
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: "smooth" });
        return;
      }
    }
    window.scrollTo({ top: 0 });
  }, [hash]);

  return <div className="animate-page-enter">{children}</div>;
}
