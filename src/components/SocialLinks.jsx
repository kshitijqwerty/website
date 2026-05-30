import {
  Github,
  Linkedin,
  Mail,
  Globe,
} from "lucide-react";

// Dev.to SVG icon (not in lucide-react)
const DevTo = ({ size = 24 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path d="M7.42 10.05c-.18-.16-.46-.23-.84-.23H6l.02 2.44.04 2.45.56-.02c.41 0 .63-.07.83-.26.24-.24.26-.36.26-2.2 0-1.91-.02-1.96-.29-2.18zM0 4.94v14.12h24V4.94H0zM8.56 15.3c-.44.58-1.06.77-2.53.77H4.71V8.53h1.4c1.67 0 2.16.18 2.6.9.27.43.29.6.32 2.57.05 2.23-.02 2.73-.47 3.3zm5.09-5.47h-2.47v1.77h1.52v1.28l-.72.04-.75.03v1.77l1.22.03 1.2.04v1.28h-1.6c-1.53 0-1.6-.01-1.87-.3l-.3-.28v-3.16c0-3.02.01-3.18.25-3.48.23-.31.25-.31 1.88-.31h1.64v1.28zm4.68 5.45c-.17.43-.64.79-1 .79-.18 0-.45-.15-.67-.39-.32-.32-.45-.63-.82-2.08l-.9-3.39-.45-1.67h.76c.44 0 .79.2.55.4l.3 1.28c.2.8.45 1.88.56 2.38.1.5.25.9.33.9.08 0 .25-.4.4-.9.14-.49.39-1.58.56-2.38l.3-1.28c.1-.4.42-.6.88-.6h.7l-.1.43c-.06.24-.32 1.1-.59 1.92-.27.82-.69 2.14-.93 2.93-.25.79-.5 1.54-.57 1.66z"/>
  </svg>
);

export default function SocialLinks() {
  const links = [
    {
      href: "https://github.com/kshitijqwerty",
      icon: Github,
      label: "GitHub",
    },
    {
      href: "https://linkedin.com/in/horz7",
      icon: Linkedin,
      label: "LinkedIn",
    },
    {
      href: "https://dev.to/kgup",
      icon: DevTo,
      label: "Dev.to",
    },
    {
      href: "https://kgup.me",
      icon: Globe,
      label: "Website",
    },
    {
      href: "mailto:kg747@proton.me",
      icon: Mail,
      label: "Email",
    },
  ];

return (
    <div className="flex gap-6 mt-8">
      {links.map(({ href, icon: Icon, label }) => (
        
         <a key={label}
          href={href}
          target={href.startsWith("mailto") ? undefined : "_blank"}
          rel={href.startsWith("mailto") ? undefined : "noreferrer"}
          aria-label={label}
          className="text-neutral-400 hover:text-white transition-colors"
        >
          <Icon size={24} />
        </a>
      ))}
    </div>
  );
}