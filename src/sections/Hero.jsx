import { motion } from "framer-motion";

import Button from "../components/Button";
import SocialLinks from "../components/SocialLinks";

export default function Hero() {
  return (
    <section className="min-h-screen grid md:grid-cols-2 gap-16 items-center py-32">
      {/* Left Content */}
      <div>
        <p className="text-xl text-neutral-400 mb-4">
          Hello, I'm
        </p>

        <motion.h1
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-5xl md:text-7xl font-bold tracking-tight leading-tight font-heading"
        >
          Kshitij Gupta
        </motion.h1>

        <p className="mt-8 text-lg text-neutral-400 leading-relaxed max-w-2xl">
          I build production AI systems — from on-device TFLite models
        running offline on Android to Kubernetes inference clusters
        serving Fortune-500 clients on GCP and Azure.
        4 years across computer vision, generative image pipelines,
        and LLM infrastructure. Full-stack enough to own the entire
        pipeline. Currently open to remote roles worldwide.
        </p>

        <div className="flex gap-4 mt-10">
          <Button href="#projects" primary>
            View Projects
          </Button>

          <Button href="#contact">
            Contact
          </Button>
        </div>

        <SocialLinks />
      </div>

      {/* Right Image */}
      <motion.div
        className="flex justify-center md:justify-end"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="w-60 h-60 md:w-80 md:h-80 rounded-full overflow-hidden border border-neutral-800 bg-neutral-900">
          <img
            src="/profile.jpg"
            alt="Profile"
            className="w-full h-full object-cover"
          />
        </div>
      </motion.div>
    </section>
  );
}