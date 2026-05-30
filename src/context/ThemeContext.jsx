import { createContext, useContext, useEffect, useMemo, useState } from "react";

const ThemeContext = createContext();

function getPreferred() {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

export function ThemeProvider({ children }) {
  const [preference, setPreference] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("theme") || "system";
    }
    return "system";
  });

  const [systemPref, setSystemPref] = useState(getPreferred);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const handler = () => setSystemPref(mq.matches ? "light" : "dark");
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const resolved = useMemo(
    () => (preference === "system" ? systemPref : preference),
    [preference, systemPref]
  );

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("light", resolved === "light");
    localStorage.setItem("theme", preference);
  }, [resolved, preference]);

  const cycle = () =>
    setPreference((p) =>
      p === "dark" ? "light" : p === "light" ? "system" : "dark"
    );

  return (
    <ThemeContext.Provider value={{ preference, resolved, cycle }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
