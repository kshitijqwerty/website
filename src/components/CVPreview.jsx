export default function CVPreview({ data }) {
  const { header, summary, skills, experience, projects, education } = data;

  return (
    <div className="bg-neutral-900/50 backdrop-blur-sm border border-neutral-800 rounded-3xl overflow-hidden">
      {/* Header */}
      {header && (
        <div className="px-8 sm:px-10 md:px-14 pt-10 sm:pt-12 md:pt-14 pb-6 text-center border-b border-neutral-800/60">
          <h3 className="text-3xl sm:text-4xl font-bold font-heading tracking-tight">{header.name}</h3>
          <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 mt-3 text-sm text-neutral-400">
            {header.links?.map((link, i) => (
              <span key={i}>
                {i > 0 && <span className="mr-3 text-neutral-700">/</span>}
                <a
                  href={link.url}
                  className="hover:text-indigo-400 transition-colors"
                  target="_blank"
                  rel="noreferrer"
                >
                  {link.label}
                </a>
              </span>
            ))}
          </div>
          {header.contact && (
            <p className="text-sm text-neutral-500 mt-2">{header.contact}</p>
          )}
        </div>
      )}

      <div className="px-8 sm:px-10 md:px-14 py-8 space-y-8">
        {/* Summary */}
        {summary && (
          <div className="text-neutral-300 leading-relaxed">{summary}</div>
        )}

        {/* Skills */}
        {skills?.length > 0 && (
          <section>
            <h4 className="inline-flex items-center gap-2 text-sm font-bold tracking-wider uppercase mb-5 text-neutral-100">
              <span className="w-1 h-4 bg-indigo-400 rounded-full" />
              Skills
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2.5">
              {skills.map((s) => (
                <div key={s.category} className="flex items-baseline gap-2 text-sm">
                  <span className="font-semibold text-neutral-100 whitespace-nowrap min-w-[120px]">
                    {s.category}
                  </span>
                  <span className="text-neutral-400">{s.items}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Experience */}
        {experience?.length > 0 && (
          <section>
            <h4 className="inline-flex items-center gap-2 text-sm font-bold tracking-wider uppercase mb-5 text-neutral-100">
              <span className="w-1 h-4 bg-indigo-400 rounded-full" />
              Experience
            </h4>
            <div className="space-y-6">
              {experience.map((exp, i) => (
                <div key={i} className="border-l-2 border-indigo-400/30 pl-4 sm:pl-5">
                  <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5">
                    <div className="flex items-baseline gap-2 flex-wrap">
                      <span className="font-semibold text-neutral-100">{exp.company}</span>
                      {exp.title && (
                        <span className="text-neutral-400 text-sm">— {exp.title}</span>
                      )}
                    </div>
                    <span className="text-xs text-neutral-500 whitespace-nowrap">
                      {exp.location}{exp.location && exp.dates ? " — " : ""}{exp.dates}
                    </span>
                  </div>
                  {exp.bullets?.length > 0 && (
                    <ul className="mt-2 space-y-1 list-disc pl-5 text-sm text-neutral-400 marker:text-neutral-600">
                      {exp.bullets.map((b, j) => (
                        <li key={j} className="leading-relaxed">{b}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Projects */}
        {projects?.length > 0 && (
          <section>
            <h4 className="inline-flex items-center gap-2 text-sm font-bold tracking-wider uppercase mb-5 text-neutral-100">
              <span className="w-1 h-4 bg-indigo-400 rounded-full" />
              Projects
            </h4>
            <div className="space-y-6">
              {projects.map((proj, i) => (
                <div key={i} className="border-l-2 border-indigo-400/30 pl-4 sm:pl-5">
                  <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5">
                    <div className="flex items-baseline gap-2 flex-wrap">
                      <span className="font-semibold text-neutral-100">{proj.name}</span>
                      {proj.github && (
                        <a
                          href={proj.github}
                          className="text-xs text-indigo-400/80 hover:text-indigo-400 transition-colors"
                          target="_blank"
                          rel="noreferrer"
                        >
                          ↗ GitHub
                        </a>
                      )}
                    </div>
                    {proj.tech && (
                      <span className="text-xs text-neutral-500 italic">{proj.tech}</span>
                    )}
                  </div>
                  {proj.bullets?.length > 0 && (
                    <ul className="mt-2 space-y-1 list-disc pl-5 text-sm text-neutral-400 marker:text-neutral-600">
                      {proj.bullets.map((b, j) => (
                        <li key={j} className="leading-relaxed">{b}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Education */}
        {education?.length > 0 && (
          <section>
            <h4 className="inline-flex items-center gap-2 text-sm font-bold tracking-wider uppercase mb-5 text-neutral-100">
              <span className="w-1 h-4 bg-indigo-400 rounded-full" />
              Education
            </h4>
            <div className="space-y-3">
              {education.map((edu, i) => (
                <div key={i} className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 text-sm">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <span className="font-semibold text-neutral-100">{edu.school}</span>
                    {edu.degree && (
                      <span className="text-neutral-400">— {edu.degree}</span>
                    )}
                  </div>
                  {edu.dates && (
                    <span className="text-xs text-neutral-500 whitespace-nowrap">{edu.dates}</span>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
