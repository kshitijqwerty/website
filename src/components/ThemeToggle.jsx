import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "../context/ThemeContext";

export default function ThemeToggle() {
  const { preference, cycle } = useTheme();

  const next = preference === "dark" ? "light" : preference === "light" ? "system" : "dark";

  const icon =
    preference === "dark" ? (
      <Moon size={16} />
    ) : preference === "light" ? (
      <Sun size={16} />
    ) : (
      <Monitor size={16} />
    );

  return (
    <button
      onClick={cycle}
      aria-label={`Theme: ${preference}. Switch to ${next}.`}
      className="relative flex items-center justify-center w-8 h-8 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
    >
      {icon}
    </button>
  );
}
