import { motion } from "framer-motion";
import { useState } from "react";
import { Github, Linkedin, Mail, Server, Brain, Database } from "lucide-react";
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

function Home() {
  const [showCV, setShowCV] = useState(false);
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 md:px-16">
      {/* Hero */}
      <section className="py-24 grid gap-6">
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
      </section>

      {/* Expertise */}
      <section className="py-20">
        <h2 className="text-3xl font-semibold mb-10">Core Expertise</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="rounded-2xl border border-neutral-800 p-6 bg-neutral-900">
            <Brain className="mb-4" />
            <h3 className="text-xl font-medium">ML Engineering</h3>
            <p className="text-neutral-400 mt-2">
              Building, training, and deploying machine learning models into production systems. Optimizing ML models for edge systems.
            </p>
            <p className="text-sm text-neutral-500 mt-4">PyTorch, TensorFlow, scikit-learn, ONNX, TF-Lite, MLflow</p>
          </div>
          <div className="rounded-2xl border border-neutral-800 p-6 bg-neutral-900">
            <Server className="mb-4" />
            <h3 className="text-xl font-medium">Backend Engineering</h3>
            <p className="text-neutral-400 mt-2">
              Designing REST APIs, microservices, and background workers.
            </p>
            <p className="text-sm text-neutral-500 mt-4">Python, FastAPI, Django, Node.js, Express</p>
          </div>
          
          <div className="rounded-2xl border border-neutral-800 p-6 bg-neutral-900">
            <Database className="mb-4" />
            <h3 className="text-xl font-medium">Databases & Infrastructure</h3>
            <p className="text-neutral-400 mt-2">
              Data modeling, performance optimization, caching, and cloud deployment.
            </p>
            <p className="text-sm text-neutral-500 mt-4">PostgreSQL, MongoDB, Redis, Docker, Kubernetes</p>
          </div>
          
        </div>
      </section>

      {/* Architecture */}
      <section className="py-20">
        <h2 className="text-3xl font-semibold mb-10">Architectural Design</h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="rounded-2xl border border-neutral-800 p-6 bg-neutral-900">
            <h3 className="text-xl font-medium mb-2">Backend System Architecture</h3>
            <p className="text-neutral-400">
              Typical architecture I design includes API gateways, stateless backend services,
              async workers, caching layers, and relational or NoSQL databases. Services are
              containerized and deployed using CI/CD pipelines.
            </p>
            <p className="text-sm text-neutral-500 mt-4">
              API Gateway → Backend Services → Cache / DB → Async Workers
            </p>
          </div>
          <div className="rounded-2xl border border-neutral-800 p-6 bg-neutral-900">
            <h3 className="text-xl font-medium mb-2">ML System Architecture</h3>
            <p className="text-neutral-400">
              End-to-end ML systems including data ingestion, feature engineering, model training,
              experiment tracking, and scalable inference via APIs or batch jobs.
            </p>
            <p className="text-sm text-neutral-500 mt-4">
              Data Sources → ETL → Training → Model Registry → Inference API
            </p>
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      {/* <section className="py-20">
        <h2 className="text-3xl font-semibold mb-10">Tech Stack</h2>
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
        <h2 className="text-3xl font-semibold mb-10">Selected Projects</h2>
        <div className="grid md:grid-cols-2 gap-8">
          {[
            {
              title: "Scalable API Platform",
              desc: "High-performance backend system handling authentication, payments, and real-time data.",
              tech: "Python, FastAPI, PostgreSQL, Docker"
            },
            {
              title: "ML Prediction Service",
              desc: "Machine learning pipeline deployed as an API for real-time inference.",
              tech: "PyTorch, MLflow, AWS, REST API"
            },
            {
              title: "Node.js Microservices",
              desc: "Event-driven microservices architecture with message queues and caching.",
              tech: "Node.js, NestJS, Redis, RabbitMQ"
            },
            {
              title: "Data Processing Pipeline",
              desc: "Large-scale data ingestion, validation, and feature engineering system.",
              tech: "Python, Pandas, Airflow"
            }
          ].map((p, i) => (
            <motion.div
              key={i}
              whileHover={{ scale: 1.03 }}
              className="rounded-2xl border border-neutral-800 p-6 bg-neutral-900"
            >
              <h3 className="text-xl font-medium">{p.title}</h3>
              <p className="text-neutral-400 mt-2">{p.desc}</p>
              <p className="text-sm text-neutral-500 mt-4">{p.tech}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CV */}
      <section id="cv" className="py-20">
        <h2 className="text-3xl font-semibold mb-6">CV</h2>
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
            href="/cv.pdf"
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
              src="/cv.pdf"
              title="CV Preview"
              className="w-full h-[800px]"
            />
          </motion.div>
        )}
      </section>

      {/* About */}
      <section className="py-20 max-w-3xl">
        <h2 className="text-3xl font-semibold mb-6">About Me</h2>
        <p className="text-neutral-400 leading-relaxed">
          I’m a backend-focused software engineer with a strong foundation in Python and Node.js.
          My work sits at the intersection of backend systems and machine learning — from designing
          robust APIs to deploying ML models that solve real-world problems at scale.
        </p>
      </section>

      {/* Contact */}
      <section id="contact" className="py-20">
        <h2 className="text-3xl font-semibold mb-6">Get in Touch</h2>
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
