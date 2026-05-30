import { Link } from "react-router-dom";
import { motion } from "framer-motion";

import list from "../data/blogList.json";

export default function BlogIndex() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <main className="max-w-6xl mx-auto px-6 md:px-10 py-20">
        <Link
          to="/"
          className="text-sm text-neutral-400 hover:text-white transition-colors inline-block mb-12"
        >
          ← Back to Home
        </Link>

        <h1 className="text-4xl md:text-5xl font-bold font-heading mb-4">
          All Posts
        </h1>

        <p className="text-neutral-400 mb-16 max-w-2xl text-lg">
          Thoughts on backend engineering, machine learning, and scalable systems.
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {list.map((post) => (
            <motion.div
              key={post.slug}
              whileHover={{ y: -4 }}
              className="rounded-3xl border border-neutral-800 bg-neutral-900 p-7 flex flex-col"
            >
              <div className="text-sm text-neutral-400 mb-4">
                <time>{post.date}</time>
              </div>

              <h2 className="text-xl font-semibold font-heading leading-snug">
                {post.title}
              </h2>

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
      </main>
    </div>
  );
}
