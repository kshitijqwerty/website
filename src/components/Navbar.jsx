import { Link, useLocation } from "react-router-dom";

import ThemeToggle from "./ThemeToggle";

export default function Navbar() {
  const { pathname, hash } = useLocation();
  const isHome = pathname === "/";

  const linkDefs = [
    { href: isHome ? "#" : "/", label: "Home" },
    { href: isHome ? "#blog" : "/#blog", label: "Blogs" },
    { href: isHome ? "#projects" : "/#projects", label: "Projects" },
    { href: isHome ? "#cv" : "/#cv", label: "CV" },
    { href: isHome ? "#contact" : "/#contact", label: "Contact" },
  ];

  const linkClass = "text-sm font-medium text-neutral-300 hover:text-white transition-colors";

  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>

      <div className="fixed top-5 left-1/2 z-50 -translate-x-1/2">
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
              >
                {label}
              </Tag>
            );
          })}

          <span className="nav-sep w-px h-5 bg-white/10" />

          <ThemeToggle />
        </nav>
      </div>
    </>
  );
}