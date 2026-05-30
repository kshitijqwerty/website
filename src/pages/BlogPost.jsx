import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { useTheme } from "../context/ThemeContext";
import ReadingProgress from "../components/ReadingProgress";
import Lightbox from "../components/Lightbox";
import posts from "../data/blogPosts.json";

function estimateReadTime(html) {
  const text = html.replace(/<[^>]+>/g, "");
  const words = text.split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(words / 200));
}

function useCopyButtons(containerRef, theme) {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const pres = container.querySelectorAll("pre:not(.mermaid-pre)");
    const buttons = [];

    const headers = [];

    pres.forEach((pre) => {
      pre.classList.add("relative", "rounded-xl", "overflow-x-auto");

      const header = document.createElement("div");
      header.className =
        "sticky left-0 z-10 flex items-center justify-between px-4 py-1.5 bg-neutral-900 border-b border-neutral-800";

      const lang = pre.getAttribute("data-language") || "";
      const label = document.createElement("span");
      label.className =
        "text-[0.75em] uppercase tracking-wide text-neutral-500";
      label.textContent = lang;

      const btn = document.createElement("button");
      btn.className =
        "flex items-center justify-center w-6 h-6 max-sm:w-8 max-sm:h-8 rounded-md text-neutral-500 hover:text-white hover:bg-neutral-700/60 transition-colors";
      btn.setAttribute("aria-label", "Copy code");
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;

      async function copyText(text) {
        if (navigator.clipboard) {
          try {
            await navigator.clipboard.writeText(text);
            return;
          } catch { /* fall through */ }
        }
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {
          const ok = document.execCommand("copy");
          if (!ok) throw new Error("execCommand copy failed");
        } finally {
          document.body.removeChild(ta);
        }
      }

      function iconHTML(paths, color) {
        return `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${paths}</svg>`;
      }

      const clipboardIcon = `<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>`;

      btn.addEventListener("click", async () => {
        const code = pre.querySelector("code");
        if (!code) return;
        try {
          await copyText(code.textContent);
          btn.innerHTML = iconHTML(`<polyline points="20 6 9 17 4 12"/>`, "#22c55e");
          setTimeout(() => {
            btn.innerHTML = iconHTML(clipboardIcon, "currentColor");
          }, 2000);
        } catch {
          btn.innerHTML = iconHTML(`<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>`, "#ef4444");
        }
      });

      header.appendChild(label);
      header.appendChild(btn);
      pre.insertBefore(header, pre.firstChild);
      headers.push(header);
      buttons.push(btn);
    });

    return () => {
      buttons.forEach((btn) => btn.remove());
      headers.forEach((h) => h.remove());
      pres.forEach((pre) => {
        pre.classList.remove("rounded-xl", "overflow-x-auto");
      });
    };
  }, [containerRef, theme]);
}

export default function BlogPost() {
  const { slug } = useParams();
  const { resolved: theme } = useTheme();
  const contentRef = useRef(null);

  const post = posts[slug];
  const [lightboxSrc, setLightboxSrc] = useState(null);

  useCopyButtons(contentRef, theme);

  useEffect(() => {
    const container = contentRef.current;
    if (!container) return;
    const imgs = container.querySelectorAll("img");
    function onClick(e) {
      setLightboxSrc(e.currentTarget.src);
    }
    imgs.forEach((img) => {
      img.addEventListener("click", onClick);
    });
    return () => {
      imgs.forEach((img) => {
        img.removeEventListener("click", onClick);
      });
    };
  }, [post, lightboxSrc]);

  useEffect(() => {
    const diagrams = contentRef.current?.querySelectorAll(".mermaid");
    if (!diagrams?.length) return;

    const isLight = theme === "light";

    import("mermaid").then(({ default: mermaid }) => {
      mermaid.initialize({
        theme: isLight ? "neutral" : "dark",
        themeVariables: isLight
          ? {
              background: "transparent",
              primaryColor: "#e5e5e5",
              primaryTextColor: "#404040",
              primaryBorderColor: "#d4d4d4",
              lineColor: "#a3a3a3",
              secondaryColor: "#f5f5f5",
              tertiaryColor: "#ffffff",
            }
          : {
              background: "transparent",
              primaryColor: "#1e293b",
              primaryTextColor: "#e2e8f0",
              primaryBorderColor: "#334155",
              lineColor: "#64748b",
              secondaryColor: "#0f172a",
              tertiaryColor: "#1e293b",
            },
      });
      mermaid.run({ nodes: diagrams });
    });
  }, [post, theme]);

  useEffect(() => {
    document.title = post
      ? `${post.title} — Kshitij Gupta`
      : "Post not found — Kshitij Gupta";

    return () => {
      document.title = "Kshitij Gupta";
    };
  }, [post]);

  if (!post) {
    return (
      <div className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center px-6">
        {lightboxSrc && (
          <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
        )}
        <div className="text-center">
          <h1 className="text-4xl font-bold font-heading mb-4">Post not found</h1>
          <p className="text-neutral-400 mb-8">
            The blog post you're looking for doesn't exist.
          </p>
          <Link
            to="/blog"
            className="btn-primary bg-white text-black px-6 py-3 rounded-2xl font-medium hover:opacity-90 transition-opacity inline-block"
          >
            Browse all posts
          </Link>
        </div>
      </div>
    );
  }

  const readTime = estimateReadTime(post.html);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 md:px-10 py-20">
      {lightboxSrc && (
        <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
      )}
      <ReadingProgress articleRef={contentRef} />
      <article className="max-w-4xl mx-auto">
        <nav className="flex items-center gap-4 text-sm text-neutral-500 mb-12">
          <Link
            to="/blog"
            className="text-neutral-400 hover:text-white transition-colors"
          >
            ← All Posts
          </Link>

          <span aria-hidden="true">·</span>

          <Link
            to="/"
            className="text-neutral-400 hover:text-white transition-colors"
          >
            Home
          </Link>
        </nav>

        <header>
          <div className="flex flex-wrap items-center gap-3 text-sm text-neutral-500 mb-4">
            <time>{post.date}</time>
            <span aria-hidden="true">·</span>
            <span>{readTime} min read</span>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold leading-tight font-heading">
            {post.title}
          </h1>
        </header>

        <hr className="border-neutral-800 my-10" />

        <div
          ref={contentRef}
          className="article-content text-lg"
          dangerouslySetInnerHTML={{ __html: post.html }}
        />

        <hr className="border-neutral-800 my-12" />

        <nav className="flex items-center gap-4 text-sm">
          <Link
            to="/blog"
            className="text-neutral-400 hover:text-white transition-colors"
          >
            ← All Posts
          </Link>

          <span aria-hidden="true" className="text-neutral-600">·</span>

          <Link
            to="/"
            className="text-neutral-400 hover:text-white transition-colors"
          >
            Home
          </Link>
        </nav>
      </article>
    </div>
  );
}