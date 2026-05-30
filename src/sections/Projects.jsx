import { useState } from "react";

import ProjectCard from "../components/ProjectCard";
import SectionTitle from "../components/SectionTitle";

import { projects } from "../data/projects";

export default function Projects() {
  const [activeTag, setActiveTag] = useState("All");

  return (
    <section id="projects" className="py-24">
      <SectionTitle
        title="Projects"
        subtitle="Selected machine learning and AI engineering projects."
      />

      {/* Filter Tags */}
      <div className="flex flex-wrap gap-3 mb-12">
        {["All", "ML", "LLM", "CV", "MLOps", "Frontend"].map(
          (tag) => (
            <button
              key={tag}
              onClick={() => setActiveTag(tag)}
              className={`px-4 py-2 rounded-full text-sm border transition-colors ${
                activeTag === tag
                  ? "tag-active bg-white text-black border-white"
                  : "bg-neutral-900 text-neutral-300 border-neutral-700 hover:border-neutral-500"
              }`}
            >
              {tag}
            </button>
          )
        )}
      </div>

      {/* Projects Grid */}
      <div className="grid md:grid-cols-2 gap-8">
        {projects
          .filter(
            (project) =>
              activeTag === "All" ||
              project.tags.includes(activeTag)
          )
          .map((project, index) => (
            <ProjectCard
              key={index}
              project={project}
            />
          ))}
      </div>
    </section>
  );
}