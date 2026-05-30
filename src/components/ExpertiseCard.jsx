export default function ExpertiseCard({
  icon: Icon,
  title,
  description,
  tech,
}) {
  return (
    <div
      className="rounded-3xl border border-neutral-800 bg-neutral-900 p-8 flex flex-col items-center text-center card-hover"
    >
      <div className="mb-5">
        <Icon size={38} />
      </div>

      <h3 className="text-xl font-semibold font-heading">
        {title}
      </h3>

      <p className="text-neutral-400 mt-4 leading-relaxed">
        {description}
      </p>

      <p className="text-sm text-neutral-400 mt-6">
        {tech}
      </p>
    </div>
  );
}