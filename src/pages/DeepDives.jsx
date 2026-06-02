import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { marked } from "marked";
import { markedHighlight } from "marked-highlight";
import hljs from "highlight.js";
import "highlight.js/styles/atom-one-dark.css";

import { useTheme } from "../context/useTheme";
import Lightbox from "../components/Lightbox";
import { deepDives } from "../data/deepDives";

marked.use(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return code;
    },
  })
);



function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^\w]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function useCopyButtons(containerRef) {
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

      const codeEl = pre.querySelector("code");
      const langMatch = (codeEl?.className || "").match(/language-(\w+)/);
      const lang = langMatch ? langMatch[1] : "";
      const label = document.createElement("span");
      label.className =
        "text-[0.75em] uppercase tracking-wide text-neutral-400";
      label.textContent = lang;

      const btn = document.createElement("button");
      btn.className =
        "flex items-center justify-center w-6 h-6 max-sm:w-8 max-sm:h-8 rounded-md text-neutral-400 hover:text-white hover:bg-neutral-700/60 transition-colors";
      btn.setAttribute("aria-label", "Copy code");
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;

      async function copyText(text) {
        if (navigator.clipboard) {
          try {
            await navigator.clipboard.writeText(text);
            return;
          } catch {}
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
          btn.innerHTML = iconHTML(
            `<polyline points="20 6 9 17 4 12"/>`,
            "#22c55e"
          );
          setTimeout(() => {
            btn.innerHTML = iconHTML(clipboardIcon, "currentColor");
          }, 2000);
        } catch {
          btn.innerHTML = iconHTML(
            `<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>`,
            "#ef4444"
          );
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
  }, [containerRef]);
}

export default function DeepDives() {
  const { slug } = useParams();
  const { resolved: theme } = useTheme();
  const contentRef = useRef(null);
  const [mdHtml, setMdHtml] = useState("");
  const [toc, setToc] = useState([]);
  const [activeId, setActiveId] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [lightboxSrc, setLightboxSrc] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const entry = deepDives.find((d) => d.slug === slug);

  useEffect(() => {
    if (!entry) {
      setLoading(false);
      setError("Deep dive not found");
      return;
    }

    setLoading(true);
    setError(null);

    fetch(`/learning/${slug}.md`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load deep dive");
        return res.text();
      })
      .then((raw) => {
        const mermaidBlocks = [];
        const preprocessed = raw.replace(
          /```mermaid\n([\s\S]*?)\n```/g,
          (_, code) => {
            const id = `<!--MERMAID_${mermaidBlocks.length}-->`;
            mermaidBlocks.push(code.trim());
            return id;
          }
        );

        let html = marked.parse(preprocessed, { gfm: true });

        html = html.replace(/<!--MERMAID_(\d+)-->/g, (_, idx) => {
          const code = mermaidBlocks[Number(idx)];
          const escaped = code
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
          return `<pre class="mermaid-pre"><div class="mermaid">\n${escaped}\n</div></pre>`;
        });

        const counter = {};
        const tocItems = [];

        html = html.replace(
          /<h([2-3])(\s[^>]*)?>(.*?)<\/h[2-3]>/gs,
          (match, level, attrs, content) => {
            const plainText = content.replace(/<[^>]+>/g, "");
            let id = slugify(plainText);
            if (counter[id]) {
              counter[id]++;
              id = `${id}-${counter[id]}`;
            } else {
              counter[id] = 1;
            }
            tocItems.push({ depth: Number(level), text: plainText, id });
            return `<h${level} id="${id}">${content}</h${level}>`;
          }
        );

        setToc(tocItems);
        setMdHtml(html);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [slug, entry]);

  useEffect(() => {
    const container = contentRef.current;
    if (!container) return;
    const headings = container.querySelectorAll("h2, h3");
    if (!headings.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
            break;
          }
        }
      },
      { rootMargin: "-80px 0px -80% 0px" }
    );

    headings.forEach((h) => observer.observe(h));
    return () => observer.disconnect();
  }, [mdHtml]);

  useEffect(() => {
    const container = contentRef.current;
    if (!container?.querySelector(".mermaid")) return;

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

      requestAnimationFrame(() => {
        mermaid.run({ querySelector: ".mermaid" }).catch((err) => {
          console.error("Mermaid render error:", err);
        });
      });
    });
  }, [mdHtml, theme]);

  useEffect(() => {
    const container = contentRef.current;
    if (!container) return;
    const imgs = container.querySelectorAll("img");
    const onClick = (e) => setLightboxSrc(e.currentTarget.src);
    imgs.forEach((img) => img.addEventListener("click", onClick));
    return () =>
      imgs.forEach((img) => img.removeEventListener("click", onClick));
  }, [mdHtml, lightboxSrc]);

  useCopyButtons(contentRef);

  useEffect(() => {
    setSidebarOpen(false);
  }, [slug]);

  useEffect(() => {
    document.title = entry
      ? `${entry.title} — Kshitij Gupta`
      : "Deep Dives — Kshitij Gupta";
    return () => {
      document.title = "Kshitij Gupta";
    };
  }, [entry]);

  if (loading) {
    return (
      <div className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center px-6">
        <p className="text-neutral-400 animate-pulse">Loading…</p>
      </div>
    );
  }

  if (!entry || error) {
    return (
      <div className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center px-6">
        {lightboxSrc && (
          <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
        )}
        <div className="text-center">
          <h1 className="text-4xl font-bold font-heading mb-4">
            {error || "Deep dive not found"}
          </h1>
          <p className="text-neutral-400 mb-8">
            The deep dive you&apos;re looking for doesn&apos;t exist.
          </p>
          <Link
            to="/learning"
            className="btn-primary bg-white text-black px-6 py-3 rounded-2xl font-medium hover:opacity-90 transition-opacity inline-block"
          >
            Browse all deep dives
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {lightboxSrc && (
        <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
      )}

      <div className="max-w-7xl mx-auto px-6 md:px-10 py-20">
        <nav className="flex items-center gap-4 text-sm text-neutral-400 mb-10">
          <Link
            to="/learning"
            className="text-neutral-400 hover:text-white transition-colors"
          >
            ← Deep Dives
          </Link>
          <span aria-hidden="true">·</span>
          <Link
            to="/"
            className="text-neutral-400 hover:text-white transition-colors"
          >
            Home
          </Link>
        </nav>

        <button
          onClick={() => setSidebarOpen((o) => !o)}
          className="lg:hidden flex items-center gap-2 text-sm text-neutral-400 hover:text-white mb-8 transition-colors"
          aria-label="Toggle table of contents"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
          Contents
        </button>

        <div className="flex gap-12 relative">
          <aside
            className={`
              lg:sticky lg:top-24 lg:block lg:w-64 lg:flex-shrink-0 lg:max-h-[calc(100vh-8rem)] lg:overflow-y-auto toc-sidebar
              ${sidebarOpen ? "block" : "hidden"}
              w-full mb-8 lg:mb-0
            `}
          >
            <nav aria-label="Table of contents">
              <h2 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider mb-4">
                Contents
              </h2>
              <ul className="space-y-1">
                {toc.map((item) => (
                  <li key={item.id}>
                    <a
                      href={`#${item.id}`}
                      onClick={(e) => {
                        e.preventDefault();
                        document
                          .getElementById(item.id)
                          ?.scrollIntoView({ behavior: "smooth" });
                        setSidebarOpen(false);
                      }}
                      className={`
                        block text-sm py-1 transition-colors
                        ${item.depth === 3 ? "pl-4" : ""}
                        ${
                          activeId === item.id
                            ? "text-emerald-400 font-medium"
                            : "text-neutral-400 hover:text-neutral-200"
                        }
                      `}
                    >
                      {item.text}
                    </a>
                  </li>
                ))}
              </ul>
            </nav>
          </aside>

          <article className="flex-1 min-w-0">
            <header className="mb-10">
              <h1 className="text-4xl md:text-5xl font-bold leading-tight font-heading">
                {entry.title}
              </h1>
              <p className="text-neutral-400 mt-4 text-lg max-w-3xl">
                {entry.description}
              </p>
            </header>

            <div
              ref={contentRef}
              className="article-content text-lg"
              dangerouslySetInnerHTML={{ __html: mdHtml }}
            />
          </article>
        </div>
      </div>
    </div>
  );
}
