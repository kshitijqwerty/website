import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

import ThemeToggle from "./ThemeToggle";

export default function Navbar() {
  const { pathname, hash } = useLocation();
  const isHome = pathname === "/";
  const [menuOpen, setMenuOpen] = useState(false);

  const linkDefs = [
    { href: isHome ? "#" : "/", label: "Home" },
    { href: "/learning", label: "Learning" },
    { href: isHome ? "#blog" : "/#blog", label: "Blogs" },
    { href: isHome ? "#projects" : "/#projects", label: "Projects" },
    { href: isHome ? "#cv" : "/#cv", label: "CV" },
    { href: isHome ? "#contact" : "/#contact", label: "Contact" },
  ];

  const linkClass = "text-sm font-medium text-neutral-300 hover:text-white transition-colors";

  function handleNav() {
    setMenuOpen(false);
  }

  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>

      {/* Desktop nav */}
      <div className="fixed top-5 max-sm:top-3 left-1/2 z-50 -translate-x-1/2 hidden sm:block">
        <nav aria-label="Main navigation" className="nav-box flex items-center gap-6 px-6 py-3 rounded-2xl border border-white/10 bg-white/10 backdrop-blur-xl shadow-lg">
          {linkDefs.map(({ href, label }) => {
            const Tag = isHome ? "a" : Link;
            const props = isHome ? { href } : { to: href };
            return (
              <Tag
                key={label}
                {...props}
                aria-current={href.replace("/", "") === hash ? "section" : undefined}
                className={linkClass}
                onClick={handleNav}
              >
                {label}
              </Tag>
            );
          })}

          <span className="nav-sep w-px h-5 bg-white/10" />

          <ThemeToggle />
        </nav>
      </div>

      {/* Mobile nav bar */}
      <div className="fixed top-0 left-0 right-0 z-50 sm:hidden">
        <div className="flex items-center justify-between px-4 py-3 bg-neutral-950/80 backdrop-blur-xl border-b border-neutral-800">
          <span className="text-sm font-semibold text-neutral-200">
            Kshitij Gupta
          </span>

          <div className="flex items-center gap-2">
            <ThemeToggle />

            <button
              onClick={() => setMenuOpen((o) => !o)}
              aria-label={menuOpen ? "Close menu" : "Open menu"}
              aria-expanded={menuOpen}
              className="flex items-center justify-center w-8 h-8 rounded-lg text-neutral-400 hover:text-white"
            >
              {menuOpen ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="3" y1="6" x2="21" y2="6" />
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="18" x2="21" y2="18" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu overlay */}
        {menuOpen && (
          <nav
            aria-label="Mobile navigation"
            className="border-b border-neutral-800 bg-neutral-950/95 backdrop-blur-xl"
          >
            <div className="flex flex-col px-4 py-4 gap-1">
              {linkDefs.map(({ href, label }) => {
                const Tag = isHome ? "a" : Link;
                const props = isHome ? { href } : { to: href };
                const active = href.replace("/", "") === hash;
                return (
                  <Tag
                    key={label}
                    {...props}
                    aria-current={active ? "section" : undefined}
                    className={`px-4 py-3 rounded-xl text-sm font-medium transition-colors ${
                      active
                        ? "bg-neutral-800 text-white"
                        : "text-neutral-400 hover:text-white hover:bg-neutral-800/60"
                    }`}
                    onClick={handleNav}
                  >
                    {label}
                  </Tag>
                );
              })}
            </div>
          </nav>
        )}
      </div>
    </>
  );
}
