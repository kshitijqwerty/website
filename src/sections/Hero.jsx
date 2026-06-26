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

        <h1
          className="text-5xl md:text-7xl font-bold tracking-tight leading-tight font-heading animate-page-enter"
          style={{ animationDuration: "0.5s" }}
        >
          Kshitij Gupta
        </h1>

        <p className="mt-8 text-lg text-neutral-400 leading-relaxed max-w-2xl">
          I build AI that ships — from edge models to LLM inference clusters in the cloud.
        Generative AI, VLM infrastructure, production ML at scale.
        Open to remote roles worldwide and relocation opportunities.
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
      <div
        className="flex justify-center md:justify-end animate-page-enter"
        style={{ animationDuration: "0.6s", animationDelay: "0.1s" }}
      >
        <div className="w-60 h-60 md:w-80 md:h-80 rounded-full overflow-hidden border border-neutral-800 bg-neutral-900">
          <img
            src="/profile.jpg"
            alt="Kshitij Gupta portrait"
            width={320}
            height={320}
            fetchpriority="high"
            className="w-full h-full object-cover"
          />
        </div>
      </div>
    </section>
  );
}