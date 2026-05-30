export default function Button({
  children,
  href,
  primary = false,
}) {
  const styles = primary
    ? "btn-primary bg-white text-black hover:opacity-90"
    : "border border-neutral-700 hover:border-neutral-500";

  if (href) {
    return (
      <a
        href={href}
        className={`px-6 py-3 rounded-2xl transition-colors ${styles}`}
      >
        {children}
      </a>
    );
  }

  return (
    <button
      className={`px-6 py-3 rounded-2xl transition-colors ${styles}`}
    >
      {children}
    </button>
  );
}