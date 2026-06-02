import { Link } from "react-router-dom";

import { deepDives } from "../data/deepDives";

export default function DeepDivesIndex() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <main className="max-w-6xl mx-auto px-6 md:px-10 py-20">
        <Link
          to="/"
          className="text-sm text-neutral-400 hover:text-white transition-colors inline-block mb-12"
        >
          ← Home
        </Link>

        <div className="mb-14">
          <h1 className="text-5xl md:text-6xl font-bold font-heading mb-3">
            Field Notes
          </h1>
          <p className="text-neutral-400 max-w-2xl text-lg">
            Random learnings that might be useful.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {deepDives.map((dd) => (
            <Link
              key={dd.slug}
              to={`/learning/${dd.slug}`}
              className="group block rounded-3xl border border-neutral-800 bg-neutral-900/50 p-7 card-hover border-t-2 border-t-emerald-500/30 hover:border-t-emerald-400/60 transition-colors"
            >
              <h2 className="text-xl font-semibold font-heading leading-snug group-hover:text-emerald-300 transition-colors">
                {dd.title}
              </h2>
              <p className="text-neutral-400 mt-3 leading-relaxed text-sm">
                {dd.description}
              </p>
              <span className="inline-flex items-center gap-1 mt-5 text-sm text-emerald-400 group-hover:text-emerald-300 transition-colors">
                Read note →
              </span>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
