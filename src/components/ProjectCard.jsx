import { Github, ExternalLink } from "lucide-react";

export default function ProjectCard({ project }) {
  return (
    <div className="group rounded-3xl border border-neutral-800 bg-neutral-900 overflow-hidden transition-opacity flex flex-col card-hover-scale">
      {/* Image */}
      <div className="relative h-56 overflow-hidden bg-neutral-800">
        <img
          src={project.image}
          alt={project.title}
          loading="lazy"
          width={800}
          height={450}
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
        />

        {/* Hover Metrics */}
        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center text-white">
          <div className="grid gap-3 text-sm text-center">
            {project.metrics?.map((metric, idx) => (
              <span
                key={idx}
                className="px-3 py-1 rounded-full bg-white/10 backdrop-blur"
              >
                {metric}
              </span>
            ))}
          </div>
        </div>

      </div>

      {/* Content */}
      <div className="p-7 flex flex-col flex-1">
        <h3 className="text-2xl font-semibold font-heading">{project.title}</h3>

        <p className="text-neutral-400 mt-3 leading-relaxed">{project.desc}</p>

        <p className="text-sm text-neutral-400 mt-5">{project.tech}</p>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mt-5">
          {project.tags?.map((tag) => (
            <span
              key={tag}
              className="text-xs px-3 py-1 rounded-full bg-neutral-800 border border-neutral-700"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* Buttons */}
        <div className="flex gap-4 mt-auto pt-7">
          {project.demo && (
            <a
              href={project.demo}
              target="_blank"
              rel="noreferrer"
              className="btn-primary bg-white text-black flex items-center justify-center gap-2 text-sm px-4 py-2 rounded-xl hover:opacity-90 transition-opacity w-full"
            >
              <ExternalLink size={16} />
              Live Demo
            </a>
          )}

          <a
            href={project.github}
            target="_blank"
            rel="noreferrer"
            className="flex items-center justify-center gap-2 text-sm px-4 py-2 rounded-xl border border-neutral-700 hover:border-neutral-500 transition-colors w-full"
          >
            <Github size={16} />
            GitHub
          </a>
        </div>
      </div>
    </div>
  );
}
