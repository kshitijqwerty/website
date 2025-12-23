import { motion } from "framer-motion";
import { useState } from "react";
import { Github, Linkedin, Mail, Server, Brain, Database, ServerCog, BrainCircuit } from "lucide-react";
import { BrowserRouter as Router, Routes, Route, Link, useParams } from "react-router-dom";

function BlogPost() {
  const { slug } = useParams();

  const posts = {
    "scalable-backend-systems": {
      title: "Designing Scalable Backend Systems",
      date: "Jan 2025",
      content: `Scalability starts with clear separation of concerns.

In this article, I walk through API design, stateless services, caching strategies, and database optimization techniques used in real-world backend systems.`
    },
    "ml-models-production": {
      title: "Deploying ML Models to Production",
      date: "Dec 2024",
      content: `Moving from notebooks to production requires robust pipelines.

This post covers data validation, model versioning, monitoring, and serving ML models via APIs.`
    },
    "async-python-node": {
      title: "Async Processing with Python & Node.js",
      date: "Nov 2024",
      content: `Async systems improve throughput and resilience.

Here I compare Celery, BullMQ, and event-driven architectures for background processing.`
    }
  };

  const post = posts[slug];

  if (!post) {
    return <div className="p-10">Post not found.</div>;
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 md:px-16 py-20 max-w-3xl">
      <Link to="/" className="text-sm text-neutral-400">← Back</Link>
      <h1 className="text-4xl font-bold mt-6">{post.title}</h1>
      <p className="text-sm text-neutral-500 mt-2">{post.date}</p>
      <article className="mt-8 whitespace-pre-line text-neutral-300 leading-relaxed">
        {post.content}
      </article>
    </div>
  );
}
function Navbar() {
  return (
    <div className="fixed top-4 left-1/2 z-50 -translate-x-1/2">
      <nav className="flex items-center gap-6 px-6 py-3 rounded-2xl border border-white/10 bg-white/10 backdrop-blur-xl shadow-lg">
        <a href="#" className="text-sm font-medium text-neutral-300 hover:text-white">Home</a>
        <a href="#projects" className="text-sm font-medium text-neutral-300 hover:text-white">Projects</a>
        <a href="#cv" className="text-sm font-medium text-neutral-300 hover:text-white">CV</a>
        {/* <a href="#blogs" className="text-sm font-medium text-neutral-300 hover:text-white">Blog</a> */}
        <a href="#contact" className="text-sm font-medium text-neutral-300 hover:text-white">Contact</a>
      </nav>
    </div>
  );
}

function ExpertiseCard({ icon: Icon, title, description, tech }) {
  return (
    <motion.div
      className="rounded-2xl border border-neutral-800 p-6 bg-neutral-900 text-center justify-center flex flex-col items-center gap-2"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.04, y: -6 }}
      transition={{ type: "spring", stiffness: 260, damping: 20 }}
      viewport={{ once: true }}
    >
      <Icon size={36} />
      <h3 className="text-xl font-medium">{title}</h3>
      <p className="text-neutral-400 mt-2">{description}</p>
      <p className="text-sm text-neutral-500 mt-4">{tech}</p>
    </motion.div>
  );
}

function Home() {
  const [showCV, setShowCV] = useState(false);
  const [activeTag, setActiveTag] = useState("All");
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 md:px-16">
      <Navbar />
      {/* Hero */}
      <section className="py-24 grid md:grid-cols-2 gap-12 items-center">
        <div className="grid gap-6">
          <h2 className="text-xl text-neutral-400">Hello, I'm</h2>
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-6xl font-bold"
          >
            Kshitij Gupta
          </motion.h1>
          <p className="text-neutral-400 max-w-2xl">
            ML Engineer specializing in <strong>Python</strong> and <strong>Node.js</strong>, with experience building
            scalable APIs, distributed systems, and production-ready <strong>Machine Learning solutions</strong>.
          </p>
          <div className="flex gap-4">
            <a href="#projects" className="px-6 py-3 rounded-2xl bg-white text-black font-medium">View Projects</a>
            <a href="#contact" className="px-6 py-3 rounded-2xl border border-neutral-700">Contact</a>
          </div>
        </div>

        {/* Profile Image */}
        <motion.div
          className="flex justify-center md:justify-end"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <div className="w-48 h-48 md:w-64 md:h-64 rounded-full overflow-hidden border-2 border-neutral-800 bg-neutral-900">
            <img
              src="/profile.jpg"
              alt="Profile"
              className="w-full h-full object-cover"
            />
          </div>
        </motion.div>
      </section>

      {/* Expertise */}
      <section className="py-20">
        <h2 className="text-4xl font-semibold mb-10">Core Expertise</h2>
        <div className="grid md:grid-cols-3 gap-8 mb-10">
          <ExpertiseCard
            icon={Brain}
            title="ML Engineering"
            description="Building, training, and deploying machine learning models into production systems. Optimizing ML models for edge systems."
            tech="PyTorch, TensorFlow, scikit-learn, ONNX, TF-Lite, MLflow"
          />
          <ExpertiseCard
            icon={Server}
            title="Backend Engineering"
            description="Designing REST APIs, microservices, and background workers."
            tech="Python, FastAPI, Django, Node.js, Express"
          />
          <ExpertiseCard
            icon={Database}
            title="Databases & Infrastructure"
            description="Data modeling, performance optimization, caching, and cloud deployment."
            tech="PostgreSQL, MongoDB, Redis, Docker, Kubernetes"
          />
        </div>
        <div className="grid md:grid-cols-2 gap-8 text-center">
          <ExpertiseCard
            icon={ServerCog}
            title="Backend System Architecture"
            description="Typical architecture I design includes API gateways, stateless backend services, async workers, caching layers, and relational or NoSQL databases. Services are containerized and deployed using CI/CD pipelines."
            tech="API Gateway → Backend Services → Cache / DB → Async Workers"
          />
          <ExpertiseCard
            icon={BrainCircuit}
            title="ML System Architecture"
            description="End-to-end ML systems including data ingestion, feature engineering, model training, experiment tracking, and scalable inference via APIs or batch jobs."
            tech="Data Sources → ETL → Training → Model Registry → Inference API"
          />
        </div>
      </section>

      {/* Tech Stack */}
      {/* <section className="py-20">
        <h2 className="text-4xl font-semibold mb-10">Tech Stack</h2>
        <div className="flex flex-wrap gap-3">
          {["Python", "Node.js", "FastAPI", "Django", "NestJS", "PostgreSQL", "MongoDB", "Redis", "Docker", "AWS", "PyTorch", "TensorFlow", "MLflow", "Airflow"].map((tech) => (
            <span
              key={tech}
              className="px-4 py-2 rounded-full text-sm bg-neutral-800 border border-neutral-700"
            >
              {tech}
            </span>
          ))}
        </div>
      </section> */}

      {/* Blogs */}
      {/* <section id="blogs" className="py-20">
        <h2 className="text-3xl font-semibold mb-10">Blog & Writing</h2>
        <p className="text-neutral-400 mb-8 max-w-2xl">
          I write about backend engineering, system design, machine learning in production,
          and lessons learned while building real-world systems.
        </p>
        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              title: "Designing Scalable Backend Systems",
              desc: "Key architectural principles for building scalable and resilient backend services.",
              date: "Jan 2025",
              link: "#"
            },
            {
              title: "Deploying ML Models to Production",
              desc: "From notebooks to APIs: a practical guide to ML engineering workflows.",
              date: "Dec 2024",
              link: "#"
            },
            {
              title: "Async Processing with Python & Node.js",
              desc: "Handling background jobs, queues, and event-driven systems effectively.",
              date: "Nov 2024",
              link: "#"
            }
          ].map((post, i) => (
            <motion.a
              key={i}
              href={post.link}
              whileHover={{ scale: 1.03 }}
              className="block rounded-2xl border border-neutral-800 p-6 bg-neutral-900"
            >
              <p className="text-sm text-neutral-500 mb-2">{post.date}</p>
              <h3 className="text-xl font-medium mb-2">{post.title}</h3>
              <p className="text-neutral-400">{post.desc}</p>
            </motion.a>
          ))}
        </div>
      </section> */}

      {/* Projects */}
      <section id="projects" className="py-20">
        <h2 className="text-4xl font-semibold mb-6">Projects</h2>

        <div className="flex flex-wrap gap-3 mb-10">
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
          {[{
            title: "Video Hosting Platform",
            desc: "High-performance Video Platform with encrypted HLS and DASH playbacks.",
            tech: "Python, Django, FFmpeg, PostgreSQL, Docker, Kubernetes",
            image: "/projects/api-platform.png",
            link: "https://github.com/kshitijqwerty/video_hosting",
            metrics: ["Auto-scaling", "<120ms latency", "99.9% uptime"],
            tags: ["Backend", "Infrastructure"]
          }, {
            title: "ML Prediction Service",
            desc: "Machine learning inference service with auto-scaling.",
            tech: "PyTorch, MLflow, AWS",
            image: "/projects/ml-service.png",
            link: "https://github.com/yourname/ml-service",
            metrics: ["~40ms inference", "Auto-scaling", "A/B tested"],
            tags: ["ML", "Backend"]
          }]
            .filter((p) => activeTag === "All" || p.tags.includes(activeTag))
            .map((p, i) => (
              <motion.a
                key={i}
                href={p.link}
                target="_blank"
                rel="noreferrer"
                whileHover={{ scale: 1.03 }}
                className="group rounded-2xl border border-neutral-800 bg-neutral-900 overflow-hidden"
              >
                <div className="relative h-48 w-full overflow-hidden bg-neutral-800">
                  <img
                    src={p.image}
                    alt={p.title}
                    className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                    <div className="grid gap-2 text-sm text-neutral-200 text-center">
                      {p.metrics.map((m, idx) => (
                        <span key={idx} className="px-3 py-1 rounded-full bg-white/10 backdrop-blur">{m}</span>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="p-6">
                  <h3 className="text-xl font-medium">{p.title}</h3>
                  <p className="text-neutral-400 mt-2">{p.desc}</p>
                  <p className="text-sm text-neutral-500 mt-4">{p.tech}</p>

                  <div className="flex flex-wrap gap-2 mt-4">
                    {p.tags.map((t) => (
                      <span key={t} className="text-xs px-2 py-1 rounded-full bg-neutral-800 border border-neutral-700">
                        {t}
                      </span>
                    ))}
                  </div>

                  <p className="mt-4 text-sm text-indigo-400">View Project -&gt;</p>
                </div>
              </motion.a>
            ))}
        </div>
      </section>

      {/* CV */}
      <section id="cv" className="py-20">
        <h2 className="text-4xl font-semibold mb-6">CV</h2>
        <p className="text-neutral-400 mb-6 max-w-2xl">
          You can view my CV directly on this page or download a copy for offline use.
        </p>
        <div className="flex gap-4 mb-8">
          <button
            onClick={() => setShowCV(!showCV)}
            className="px-6 py-3 rounded-2xl bg-white text-black font-medium"
          >
            {showCV ? "Hide CV" : "View CV"}
          </button>
          <a
            href="/kshitij_cv.pdf"
            download
            className="px-6 py-3 rounded-2xl border border-neutral-700"
          >
            Download CV
          </a>
        </div>
        {showCV && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl overflow-hidden border border-neutral-800 bg-neutral-900"
          >
            <iframe
              src="/kshitij_cv.pdf"
              title="CV Preview"
              className="w-full h-[800px]"
            />
          </motion.div>
        )}
      </section>

      {/* About */}
      <section className="py-20 max-w-3xl">
        <h2 className="text-4xl font-semibold mb-6">About Me</h2>
        <p className="text-neutral-400 leading-relaxed">
          I’m a backend-focused software engineer with a strong foundation in Python and Node.js.
          My work sits at the intersection of backend systems and machine learning — from designing
          robust APIs to deploying ML models that solve real-world problems at scale.
        </p>
      </section>

      {/* Contact */}
      <section id="contact" className="py-20">
        <h2 className="text-4xl font-semibold mb-6">Get in Touch</h2>
        <div className="flex gap-6">
          <a href="https://github.com/kshitijqwerty" target="_blank"><Github /></a>
          <a href="https://linkedin.com/in/horz7" target="_blank"><Linkedin /></a>
          <a href="mailto:kg747@proton.me"><Mail /></a>
        </div>
      </section>

      <footer className="py-10 text-sm text-neutral-600">
        © {new Date().getFullYear()} Kshitij Gupta
      </footer>
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
