import SocialLinks from "../components/SocialLinks";

export default function Contact() {
  return (
    <section id="contact" className="py-24">
      <h2 className="text-4xl font-bold mb-6">
        Get in Touch
      </h2>

      <p className="text-neutral-400 max-w-2xl">
        I’m always interested in backend engineering,
        infrastructure, and machine learning projects.
      </p>

      <SocialLinks />
    </section>
  );
}