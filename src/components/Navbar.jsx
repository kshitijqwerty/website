export default function Navbar() {
  return (
    <div className="fixed top-5 left-1/2 z-50 -translate-x-1/2">
      <nav className="flex items-center gap-6 px-6 py-3 rounded-2xl border border-white/10 bg-white/10 backdrop-blur-xl shadow-lg">
        <a
          href="#"
          className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
        >
          Home
        </a>

        <a
          href="#projects"
          className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
        >
          Projects
        </a>

        <a
          href="#cv"
          className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
        >
          CV
        </a>

        <a
          href="#contact"
          className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
        >
          Contact
        </a>
      </nav>
    </div>
  );
}