import { motion } from "framer-motion";
import { useState } from "react";
import {
  Github,
  Linkedin,
  Mail,
  Server,
  Brain,
  Database,
  ServerCog,
  BrainCircuit,
} from "lucide-react";

import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useParams,
} from "react-router-dom";

function BlogPost() {
  const { slug } = useParams();

  const posts = {
    "scalable-backend-systems": {
      title: "Designing Scalable Backend Systems",
      date: "Jan 2025",
      content: `Scalability starts with clear separation of concerns.

In this article, I walk through API design, stateless services, caching strategies, and database optimization techniques used in real-world backend systems.`,
    },

    "ml-models-production": {
      title: "Deploying ML Models to Production",
      date: "Dec 2024",
      content: `Moving from notebooks to production requires robust pipelines.

This post covers data validation, model versioning, monitoring, and serving ML models via APIs.`,
    },

    "async-python-node": {
      title: "Async Processing with Python & Node.js",
      date: "Nov 2024",
      content: `Async systems improve throughput and resilience.

Here I compare Celery, BullMQ, and event-driven architectures for background processing.`,
    },
  };

  const post = posts[slug];

  if (!post) {
    return (
      <div className="min-h-screen bg-neutral-950 text-white p-10">
        Post not found.
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 md:px-10 py-20">
      <div className="max-w-3xl mx-auto">
        <Link
          to="/"
          className="text-sm text-neutral-400 hover:text-white transition-colors"
        >
          ← Back
        </Link>

        <h1 className="text-4xl md:text-5xl font-bold mt-6">
          {post.title}
        </h1>

        <p className="text-sm text-neutral-500 mt-3">{post.date}</p>

        <article className="mt-10 whitespace-pre-line text-neutral-300 leading-relaxed text-lg">
          {post.content}
        </article>
      </div>
    </div>
  );
}

function Navbar() {
  return (
    <div className="fixed top-5 left-1/2 z-50 -translate-x-1/2">
      <nav className="flex items-center gap-6 px-6 py-3 rounded-2xl border border-white/10 bg-white/10 backdrop-blur-xl shadow-lg">
        <a
          href="#"
          className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
        >
          Home
        </a>

        <a
          href="#projects"
          className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
        >
          Projects
        </a>

        <a
          href="#cv"
          className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
        >
          CV
        </a>

        <a
          href="#contact"
          className="text-sm font-medium text-neutral-300 hover:text-white transition-colors"
        >
          Contact
        </a>
      </nav>
    </div>
  );
}

function ExpertiseCard({ icon: Icon, title, description, tech }) {
  return (
    <motion.div
      className="rounded-3xl border border-neutral-800 bg-neutral-900 p-8 flex flex-col items-center text-center"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.03, y: -4 }}
      transition={{ type: "spring", stiffness: 260, damping: 20 }}
      viewport={{ once: true }}
    >
      <div className="mb-5">
        <Icon size={38} />
      </div>

      <h3 className="text-xl font-semibold">{title}</h3>

      <p className="text-neutral-400 mt-4 leading-relaxed">
        {description}
      </p>

      <p className="text-sm text-neutral-500 mt-6">{tech}</p>
    </motion.div>
  );
}

function Home() {
  const [showCV, setShowCV] = useState(false);
  const [activeTag, setActiveTag] = useState("All");

  const projects = [
    {
      title: "Video Hosting Platform",
      desc: "High-performance video platform with encrypted HLS and DASH playback.",
      tech: "Python, Django, FFmpeg, PostgreSQL, Docker, Kubernetes",
      image: "/projects/api-platform.png",
      link: "https://github.com/kshitijqwerty/video_hosting",
      metrics: ["Auto-scaling", "<120ms latency", "99.9% uptime"],
      tags: ["Backend", "Infrastructure"],
    },

    {
      title: "ML Prediction Service",
      desc: "Machine learning inference service with autoscaling and monitoring.",
      tech: "PyTorch, MLflow, AWS",
      image: "/projects/ml-service.png",
      link: "https://github.com/yourname/ml-service",
      metrics: ["~40ms inference", "Auto-scaling", "A/B tested"],
      tags: ["ML", "Backend"],
    },
  ];

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <Navbar />

      <div className="max-w-6xl mx-auto px-6 md:px-10">
        {/* HERO */}
        <section className="min-h-screen grid md:grid-cols-2 gap-20 items-center py-32">
          <div>
            <p className="text-xl text-neutral-400 mb-4">Hello, I'm</p>

            <motion.h1
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-5xl md:text-7xl font-bold tracking-tight leading-tight"
            >
              Kshitij Gupta
            </motion.h1>

            <p className="mt-8 text-lg text-neutral-400 leading-relaxed max-w-2xl">
              ML Engineer specializing in <strong>Python</strong> and{" "}
              <strong>Node.js</strong>, building scalable APIs,
              distributed backend systems, and production-ready{" "}
              <strong>Machine Learning solutions</strong>.
            </p>

            <div className="flex gap-4 mt-10">
              <a
                href="#projects"
                className="px-6 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 transition-opacity"
              >
                View Projects
              </a>

              <a
                href="#contact"
                className="px-6 py-3 rounded-2xl border border-neutral-700 hover:border-neutral-500 transition-colors"
              >
                Contact
              </a>
            </div>
          </div>

          {/* PROFILE IMAGE */}
          <motion.div
            className="flex justify-center md:justify-end"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
          >
            <div className="w-56 h-56 md:w-72 md:h-72 rounded-full overflow-hidden border border-neutral-800 bg-neutral-900 shadow-2xl">
              <img
                src="/profile.jpg"
                alt="Profile"
                className="w-full h-full object-cover"
              />
            </div>
          </motion.div>
        </section>

        {/* EXPERTISE */}
        <section className="py-24">
          <h2 className="text-4xl font-bold mb-14">
            Core Expertise
          </h2>

          <div className="grid md:grid-cols-3 gap-8 mb-10">
            <ExpertiseCard
              icon={Brain}
              title="ML Engineering"
              description="Building, training, and deploying machine learning systems into production environments."
              tech="PyTorch • TensorFlow • scikit-learn • ONNX • TF-Lite • MLflow"
            />

            <ExpertiseCard
              icon={Server}
              title="Backend Engineering"
              description="Designing scalable REST APIs, distributed services, and asynchronous workers."
              tech="Python • FastAPI • Django • Node.js • Express"
            />

            <ExpertiseCard
              icon={Database}
              title="Databases & Infrastructure"
              description="Data modeling, performance optimization, caching, and cloud-native deployments."
              tech="PostgreSQL • MongoDB • Redis • Docker • Kubernetes"
            />
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <ExpertiseCard
              icon={ServerCog}
              title="Backend System Architecture"
              description="API gateways, stateless services, queues, async workers, caching layers, and scalable deployments."
              tech="API Gateway → Services → Cache / DB → Async Workers"
            />

            <ExpertiseCard
              icon={BrainCircuit}
              title="ML System Architecture"
              description="End-to-end ML pipelines from ingestion and training to deployment and scalable inference."
              tech="Data → ETL → Training → Registry → Inference"
            />
          </div>
        </section>

        {/* PROJECTS */}
        <section id="projects" className="py-24">
          <h2 className="text-4xl font-bold mb-8">Projects</h2>

          <div className="flex flex-wrap gap-3 mb-12">
            {["All", "Backend", "ML", "Infrastructure"].map((tag) => (
              <button
                key={tag}
                onClick={() => setActiveTag(tag)}
                className={`px-4 py-2 rounded-full text-sm border transition-colors ${
                  activeTag === tag
                    ? "bg-white text-black border-white"
                    : "bg-neutral-900 text-neutral-300 border-neutral-700 hover:border-neutral-500"
                }`}
              >
                {tag}
              </button>
            ))}
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {projects
              .filter(
                (p) =>
                  activeTag === "All" ||
                  p.tags.includes(activeTag)
              )
              .map((p, i) => (
                <motion.a
                  key={i}
                  href={p.link}
                  target="_blank"
                  rel="noreferrer"
                  whileHover={{ scale: 1.02 }}
                  className="group rounded-3xl border border-neutral-800 bg-neutral-900 overflow-hidden"
                >
                  <div className="relative h-56 overflow-hidden bg-neutral-800">
                    <img
                      src={p.image}
                      alt={p.title}
                      className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                    />

                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                      <div className="grid gap-3 text-sm text-center">
                        {p.metrics.map((m, idx) => (
                          <span
                            key={idx}
                            className="px-3 py-1 rounded-full bg-white/10 backdrop-blur"
                          >
                            {m}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="p-7">
                    <h3 className="text-2xl font-semibold">
                      {p.title}
                    </h3>

                    <p className="text-neutral-400 mt-3 leading-relaxed">
                      {p.desc}
                    </p>

                    <p className="text-sm text-neutral-500 mt-5">
                      {p.tech}
                    </p>

                    <div className="flex flex-wrap gap-2 mt-5">
                      {p.tags.map((t) => (
                        <span
                          key={t}
                          className="text-xs px-3 py-1 rounded-full bg-neutral-800 border border-neutral-700"
                        >
                          {t}
                        </span>
                      ))}
                    </div>

                    <p className="mt-6 text-sm text-indigo-400">
                      View Project →
                    </p>
                  </div>
                </motion.a>
              ))}
          </div>
        </section>

        {/* CV */}
        <section id="cv" className="py-24">
          <h2 className="text-4xl font-bold mb-6">CV</h2>

          <p className="text-neutral-400 mb-8 max-w-2xl">
            You can preview my CV directly on the page or
            download a copy for offline viewing.
          </p>

          <div className="flex gap-4 mb-10">
            <button
              onClick={() => setShowCV(!showCV)}
              className="px-6 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 transition-opacity"
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

        {/* ABOUT */}
        <section className="py-24 max-w-3xl">
          <h2 className="text-4xl font-bold mb-6">About Me</h2>

          <p className="text-neutral-400 leading-relaxed text-lg">
            I’m a backend-focused software engineer with a
            strong foundation in Python and Node.js. My work
            sits at the intersection of backend systems and
            machine learning — from designing robust APIs to
            deploying ML models that solve real-world problems
            at scale.
          </p>
        </section>

        {/* CONTACT */}
        <section id="contact" className="py-24">
          <h2 className="text-4xl font-bold mb-6">
            Get in Touch
          </h2>

          <div className="flex gap-6 mt-8">
            <a
              href="https://github.com/kshitijqwerty"
              target="_blank"
              rel="noreferrer"
              className="text-neutral-400 hover:text-white transition-colors"
            >
              <Github size={24} />
            </a>

            <a
              href="https://linkedin.com/in/horz7"
              target="_blank"
              rel="noreferrer"
              className="text-neutral-400 hover:text-white transition-colors"
            >
              <Linkedin size={24} />
            </a>

            <a
              href="mailto:kg747@proton.me"
              className="text-neutral-400 hover:text-white transition-colors"
            >
              <Mail size={24} />
            </a>
          </div>
        </section>

        {/* FOOTER */}
        <footer className="py-10 text-sm text-neutral-600 border-t border-neutral-900">
          © {new Date().getFullYear()} Kshitij Gupta
        </footer>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/blog/:slug" element={<BlogPost />} />
      </Routes>
    </Router>
  );
}