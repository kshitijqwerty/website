import { Link } from "react-router-dom";

import SectionTitle from "../components/SectionTitle";
import list from "../data/blogList.json";

function formatDate(iso) {
  if (!iso) return "";
  return new Date(iso + "T00:00:00").toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export default function Blogs() {
  const latest = list.slice(0, 3);

  return (
    <section id="blog" className="py-24">
      <SectionTitle
        title="Blog & Writing"
        subtitle="Notes on backend engineering, machine learning, and distributed systems."
      />

      <div className="grid md:grid-cols-3 gap-6">
        {latest.map((post) => (
          <Link
            key={post.slug}
            to={`/blog/${post.slug}`}
            className="group block rounded-3xl border border-neutral-800 bg-neutral-900/50 p-7 card-hover border-t-2 border-t-violet-500/30 hover:border-t-violet-400/60 transition-colors"
          >
            <div className="text-xs text-neutral-500 mb-4">
              <time>{formatDate(post.date)}</time>
            </div>

            <h3 className="text-xl font-semibold font-heading leading-snug group-hover:text-violet-300 transition-colors">
              {post.title}
            </h3>

            <p className="text-neutral-400 mt-3 leading-relaxed text-sm">
              {post.description}
            </p>

            <span className="inline-flex items-center gap-1 mt-5 text-sm text-violet-400 group-hover:text-violet-300 transition-colors">
              Read post →
            </span>
          </Link>
        ))}
      </div>

      <div className="text-center mt-12">
        <Link
          to="/blog"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-2xl border border-neutral-700 hover:border-neutral-500 transition-colors text-sm font-medium"
        >
          View all posts →
        </Link>
      </div>
    </section>
  );
}