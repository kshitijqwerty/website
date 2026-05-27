import {
  Brain,
  Server,
  Database,
  ServerCog,
  BrainCircuit,
} from "lucide-react";

import ExpertiseCard from "../components/ExpertiseCard";
import SectionTitle from "../components/SectionTitle";

export default function Expertise() {
  return (
    <section className="py-24">
      <SectionTitle
        title="Core Expertise"
        subtitle="Computer vision, LLM systems, and end-to-end ML infrastructure."
      />

      <div className="grid md:grid-cols-3 gap-8 mb-8">
        <ExpertiseCard
          icon={Brain}
          title="ML Engineering"
          description="Production CV and GenAI systems — from on-device TFLite models to Kubernetes inference clusters serving Fortune-500 clients."
          tech="PyTorch • OpenCV • TFLite • ONNX • Triton • CLIP"
        />

        <ExpertiseCard
          icon={BrainCircuit}
          title="LLM Systems"
          description="RAG pipelines, hybrid search, LLM evaluation harnesses, and agentic workflows with CI/CD quality gates."
          tech="LangChain • Qdrant • RAGAS • Ollama • Groq • Anthropic"
        />

        <ExpertiseCard
          icon={ServerCog}
          title="MLOps / Infra"
          description="Full ML lifecycle ownership — annotation, training, versioning, and automated cloud deployment at scale."
          tech="Docker • Kubernetes • KServe • GCP • Azure • GitHub Actions"
        />
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <ExpertiseCard
          icon={Server}
          title="Full-Stack AI"
          description="End-to-end product ownership — model training to deployed API to frontend. No handoffs needed."
          tech="FastAPI • Next.js • Node.js • PostgreSQL • Redis"
        />

        <ExpertiseCard
          icon={Database}
          title="ML Architecture"
          description="Designed pipelines from CVAT annotation to KServe inference — and RAG systems from chunking to streaming citation UI."
          tech="CVAT → Training → Triton → KServe | PDF → Qdrant → Ollama → Stream"
        />
      </div>
    </section>
  );
}