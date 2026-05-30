import { motion } from "framer-motion";

export default function ExpertiseCard({
  icon: Icon,
  title,
  description,
  tech,
}) {
  return (
    <motion.div
      className="rounded-3xl border border-neutral-800 bg-neutral-900 p-8 flex flex-col items-center text-center"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.03, y: -4 }}
      transition={{
        type: "spring",
        stiffness: 260,
        damping: 20,
      }}
      viewport={{ once: true }}
    >
      <div className="mb-5">
        <Icon size={38} />
      </div>

      <h3 className="text-xl font-semibold font-heading">
        {title}
      </h3>

      <p className="text-neutral-400 mt-4 leading-relaxed">
        {description}
      </p>

      <p className="text-sm text-neutral-500 mt-6">
        {tech}
      </p>
    </motion.div>
  );
}