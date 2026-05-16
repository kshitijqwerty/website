import {
  Github,
  Linkedin,
  Mail,
} from "lucide-react";

export default function SocialLinks() {
  return (
    <div className="flex gap-6 mt-8">
      <a
        href="https://github.com/kshitijqwerty"
        target="_blank"
        rel="noreferrer"
        className="text-neutral-400 hover:text-white transition-colors"
      >
        <Github size={24} />
      </a>

      <a
        href="https://linkedin.com/in/horz7"
        target="_blank"
        rel="noreferrer"
        className="text-neutral-400 hover:text-white transition-colors"
      >
        <Linkedin size={24} />
      </a>

      <a
        href="mailto:kg747@proton.me"
        className="text-neutral-400 hover:text-white transition-colors"
      >
        <Mail size={24} />
      </a>
    </div>
  );
}