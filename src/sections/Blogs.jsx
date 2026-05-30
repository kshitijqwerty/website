import { motion } from "framer-motion";
import { Link } from "react-router-dom";

import SectionTitle from "../components/SectionTitle";
import list from "../data/blogList.json";

export default function Blogs() {
  const latest = list.slice(0, 3);

  return (
    <section id="blog" className="py-24">
      <SectionTitle
        title="Blog & Writing"
        subtitle="Thoughts on backend engineering, machine learning, and scalable systems."
      />

      <div className="grid md:grid-cols-3 gap-8">
        {latest.map((post) => (
          <motion.div
            key={post.slug}
            whileHover={{ y: -4 }}
            className="rounded-3xl border border-neutral-800 bg-neutral-900 p-7 flex flex-col"
          >
            <div className="text-sm text-neutral-400 mb-3">
              <time>{post.date}</time>
            </div>

            <h3 className="text-xl font-semibold font-heading leading-snug">
              {post.title}
            </h3>

            <p className="text-neutral-400 mt-4 leading-relaxed flex-1">
              {post.description}
            </p>

            <Link
              to={`/blog/${post.slug}`}
              className="inline-flex items-center gap-1 mt-6 text-indigo-400 hover:text-indigo-300 transition-colors"
            >
              Read Article →
            </Link>
          </motion.div>
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