export default function About() {
  const skills = [
    "Python", "PyTorch", "Computer Vision", "LLMs", "RAG",
    "FastAPI", "Docker", "Kubernetes", "GCP", "Azure",
    "LangChain", "Qdrant", "Triton", "TFLite", "ONNX", "Next.js"
  ];

  return (
    <section className="py-24 max-w-3xl">
      <h2 className="text-4xl font-bold mb-6 font-heading">About Me</h2>

      <p className="text-neutral-400 leading-relaxed text-lg mb-6">
        I'm an ML Engineer with 4 years of production experience
        across computer vision, generative AI, and LLM systems.
      </p>

      <p className="text-neutral-400 leading-relaxed text-lg mb-6">
        My work spans the full ML lifecycle — from CVAT annotation
        pipelines and model training to KServe + Triton inference
        clusters on GCP and Azure. I've shipped generative image
        pipelines for Fortune-500 clients (LG, Ford, Marks &amp; Spencer)
        and built sub-5MB on-device TFLite models for offline
        Android inference.
      </p>


      <p className="text-neutral-400 leading-relaxed text-lg mb-8">
        Masters of Technology · IIIT Hyderabad &nbsp;|&nbsp;
        Bachelors of Technology · IIIT Delhi &nbsp;|&nbsp;
        KVPY Scholar &nbsp;|&nbsp; Dean's List
      </p>

      {/* Skills */}
      <div className="flex flex-wrap gap-2 mb-8">
        {skills.map(skill => (
          <span
            key={skill}
            className="text-xs px-3 py-1 rounded-full bg-neutral-800 border border-neutral-700 text-neutral-300"
          >
            {skill}
          </span>
        ))}
      </div>

{/* Status */}
      <div className="flex items-center gap-2 text-sm text-neutral-400">
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse inline-block" />
  Open to remote roles worldwide · Building in public at&nbsp;
  
    <a href="https://dev.to/kgup"
    target="_blank"
    rel="noreferrer"
    className="text-white hover:underline">
    dev.to/kgup
  </a>
      </div>

    </section>
  );
}