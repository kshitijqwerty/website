import { Link } from "react-router-dom";

import deepDives from "../data/learningList.json";
import { useBookmarks } from "../hooks/useBookmarks";

export default function DeepDivesIndex() {
  const { isBookmarked } = useBookmarks("learning");
  const sorted = [...deepDives].sort((a, b) => {
    const aBm = isBookmarked(a.slug) ? 0 : 1;
    const bBm = isBookmarked(b.slug) ? 0 : 1;
    return aBm - bBm;
  });
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
            Random learning notes that might be useful.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sorted.map((dd) => (
            <Link
              key={dd.slug}
              to={`/learning/${dd.slug}`}
              className="group relative block rounded-3xl border border-neutral-800 bg-neutral-900/50 p-7 card-hover border-t-2 border-t-emerald-500/30 hover:border-t-emerald-400/60 transition-colors"
            >
              {isBookmarked(dd.slug) && (
                <span className="absolute top-4 right-4" aria-label="Bookmarked">
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="#34d399"
                    stroke="#34d399"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
                  </svg>
                </span>
              )}
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
