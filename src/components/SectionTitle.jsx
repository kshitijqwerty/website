export default function SectionTitle({
  title,
  subtitle,
}) {
  return (
    <div className="mb-12">
      <h2 className="text-4xl font-bold font-heading">
        {title}
      </h2>

      {subtitle && (
        <p className="text-neutral-400 mt-4 max-w-2xl">
          {subtitle}
        </p>
      )}
    </div>
  );
}