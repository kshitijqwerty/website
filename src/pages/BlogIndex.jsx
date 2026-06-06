import { Link } from "react-router-dom";

import list from "../data/blogList.json";
import { useBookmarks } from "../hooks/useBookmarks";

function formatDate(iso) {
  if (!iso) return "";
  return new Date(iso + "T00:00:00").toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export default function BlogIndex() {
  const { isBookmarked } = useBookmarks("blog");
  const sorted = [...list].sort((a, b) => {
    const aBm = isBookmarked(a.slug) ? 0 : 1;
    const bBm = isBookmarked(b.slug) ? 0 : 1;
    if (aBm !== bBm) return aBm - bBm;
    return b.date.localeCompare(a.date);
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
            Blogs
          </h1>
          <p className="text-neutral-400 max-w-2xl text-lg">
            Notes on backend engineering, machine learning, and distributed systems.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sorted.map((post) => (
            <Link
              key={post.slug}
              to={`/blog/${post.slug}`}
              className="group relative block rounded-3xl border border-neutral-800 bg-neutral-900/50 p-7 card-hover border-t-2 border-t-violet-500/30 hover:border-t-violet-400/60 transition-colors"
            >
              {isBookmarked(post.slug) && (
                <span className="absolute top-4 right-4" aria-label="Bookmarked">
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="#a78bfa"
                    stroke="#a78bfa"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
                  </svg>
                </span>
              )}
              <div className="text-xs text-neutral-500 mb-4">
                <time>{formatDate(post.date)}</time>
              </div>

              <h2 className="text-xl font-semibold font-heading leading-snug group-hover:text-violet-300 transition-colors">
                {post.title}
              </h2>

              <p className="text-neutral-400 mt-3 leading-relaxed text-sm">
                {post.description}
              </p>

              <span className="inline-flex items-center gap-1 mt-5 text-sm text-violet-400 group-hover:text-violet-300 transition-colors">
                Read post →
              </span>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
