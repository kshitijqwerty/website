import SocialLinks from "../components/SocialLinks";

export default function Contact() {
  return (
    <section id="contact" className="py-24">
      <h2 className="text-4xl font-bold mb-6 font-heading">
        Get in Touch
      </h2>

      <p className="text-neutral-400 max-w-2xl">
        I'm open to remote ML engineering roles worldwide — 
        computer vision, LLM systems, and AI infrastructure. 
        Always happy to talk shop.
      </p>

      <SocialLinks />
    </section>
  );
}