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
        subtitle="Backend engineering, machine learning infrastructure, and scalable systems."
      />

      <div className="grid md:grid-cols-3 gap-8 mb-8">
        <ExpertiseCard
          icon={Brain}
          title="ML Engineering"
          description="Building and deploying production-grade machine learning systems."
          tech="PyTorch • TensorFlow • MLflow • ONNX"
        />

        <ExpertiseCard
          icon={Server}
          title="Backend Engineering"
          description="Scalable APIs, async systems, queues, and microservices."
          tech="Python • FastAPI • Django • Node.js"
        />

        <ExpertiseCard
          icon={Database}
          title="Infrastructure"
          description="Databases, containers, orchestration, and cloud-native deployments."
          tech="PostgreSQL • Redis • Docker • Kubernetes"
        />
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <ExpertiseCard
          icon={ServerCog}
          title="Backend Architecture"
          description="Distributed systems with stateless services, async workers, and scalable APIs."
          tech="Gateway → Services → Cache → Workers"
        />

        <ExpertiseCard
          icon={BrainCircuit}
          title="ML Architecture"
          description="End-to-end ML pipelines from ingestion to inference."
          tech="ETL → Training → Registry → Deployment"
        />
      </div>
    </section>
  );
}