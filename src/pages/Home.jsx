import Hero from "../sections/Hero";
import Expertise from "../sections/Expertise";
import Projects from "../sections/Projects";
import CV from "../sections/CV";
import About from "../sections/About";
import Contact from "../sections/Contact";
import Blogs from "../sections/Blogs";

export default function Home() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <main id="main-content" className="max-w-6xl mx-auto px-6 md:px-10">
        <Hero />

        <Expertise />

        <Projects />

        <Blogs />

        <CV />

        <About />

        <Contact />

        {/* Footer */}
        <footer className="py-10 border-t border-neutral-900 text-sm text-neutral-400">
          © {new Date().getFullYear()} Kshitij Gupta
        </footer>
      </main>
    </div>
  );
}