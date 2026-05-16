import { motion } from "framer-motion";
import { Link } from "react-router-dom";

import SectionTitle from "../components/SectionTitle";

const posts = [
  {
    title: "Designing Scalable Backend Systems",
    slug: "scalable-backend-systems",
    date: "Jan 2025",
    desc: "Key architectural principles for scalable backend systems.",
  },

  {
    title: "Deploying ML Models to Production",
    slug: "ml-models-production",
    date: "Dec 2024",
    desc: "Building reliable ML systems from experimentation to deployment.",
  },

  {
    title: "Async Processing with Python & Node.js",
    slug: "async-python-node",
    date: "Nov 2024",
    desc: "Queues, workers, and event-driven backend architectures.",
  },
];

export default function Blogs() {
  return (
    <section className="py-24">
      <SectionTitle
        title="Blog & Writing"
        subtitle="Thoughts on backend engineering, machine learning, and scalable systems."
      />

      <div className="grid md:grid-cols-3 gap-8">
        {posts.map((post, index) => (
          <motion.div
            key={index}
            whileHover={{ y: -4 }}
            className="rounded-3xl border border-neutral-800 bg-neutral-900 p-7"
          >
            <p className="text-sm text-neutral-500">
              {post.date}
            </p>

            <h3 className="text-xl font-semibold mt-3">
              {post.title}
            </h3>

            <p className="text-neutral-400 mt-4 leading-relaxed">
              {post.desc}
            </p>

            <Link
              to={`/blog/${post.slug}`}
              className="inline-block mt-6 text-indigo-400 hover:text-indigo-300 transition-colors"
            >
              Read Article →
            </Link>
          </motion.div>
        ))}
      </div>
    </section>
  );
}