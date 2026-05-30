import { useState } from "react";
import { motion } from "framer-motion";

export default function CV() {
  const [showCV, setShowCV] = useState(false);

  return (
    <section id="cv" className="py-24">
      <h2 className="text-4xl font-bold mb-6 font-heading">CV</h2>

      <p className="text-neutral-400 mb-8 max-w-2xl">
        You can preview my CV directly on the page or
        download a copy for offline viewing.
      </p>

      <div className="flex gap-4 mb-10">
        <button
          onClick={() => setShowCV(!showCV)}
          className="btn-primary bg-white text-black px-6 py-3 rounded-2xl font-medium hover:opacity-90 transition-opacity"
        >
          {showCV ? "Hide CV" : "View CV"}
        </button>

        <a
          href="/kshitij_cv.pdf"
          download
          className="px-6 py-3 rounded-2xl border border-neutral-700 hover:border-neutral-500 transition-colors"
        >
          Download CV
        </a>
      </div>

      {showCV && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-3xl overflow-hidden border border-neutral-800 bg-neutral-900"
        >
          <iframe
            src="/kshitij_cv.pdf"
            title="CV Preview"
            className="w-full h-[800px]"
          />
        </motion.div>
      )}
    </section>
  );
}