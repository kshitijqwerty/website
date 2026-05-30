export default function Navbar() {
  const links = [
    { href: "#", label: "Home" },
    { href: "#projects", label: "Projects" },
    { href: "#cv", label: "CV" },
    { href: "#contact", label: "Contact" },
  ];

  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>

      <div className="fixed top-5 left-1/2 z-50 -translate-x-1/2">
        <nav aria-label="Main navigation" className="flex items-center gap-6 px-6 py-3 rounded-2xl border border-white/10 bg-white/10 backdrop-blur-xl shadow-lg">
          {links.map(({ href, label }) => (
            <a
              key={href}
              href={href}
              aria-current={href === window.location.hash ? "section" : undefined}
              className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
            >
              {label}
            </a>
          ))}
        </nav>
      </div>
    </>
  );
}