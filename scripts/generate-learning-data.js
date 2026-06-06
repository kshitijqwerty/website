import { readFileSync, readdirSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import matter from "gray-matter";
import { marked } from "marked";
import { markedHighlight } from "marked-highlight";
import hljs from "highlight.js";

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

marked.setOptions({ gfm: true, breaks: true });

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^\w]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const contentDir = join(__dirname, "..", "content", "learning");
const outDir = join(__dirname, "..", "src", "data");
const listFile = join(outDir, "learningList.json");
const postsFile = join(outDir, "learningPosts.json");

const files = readdirSync(contentDir).filter((f) => f.endsWith(".md"));

const list = [];
const posts = {};

for (const file of files) {
  const raw = readFileSync(join(contentDir, file), "utf-8");
  const { data, content } = matter(raw);

  if (data.published === false) continue;

  const slug = data.slug || file.replace(/\.md$/, "");

  let body = content.replace(/\n## Table of Contents\n[\s\S]*?(?:\n---)?(?=\n## |$)/, "");

  const mermaidBlocks = [];
  const preprocessed = body.replace(
    /```mermaid\n([\s\S]*?)\n```/g,
    (_, code) => {
      const id = `<!--MERMAID_${mermaidBlocks.length}-->`;
      mermaidBlocks.push(code.trim());
      return id;
    }
  );

  let html = marked.parse(preprocessed);

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
  function unescape(text) {
    return text
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#x27;/g, "'")
      .replace(/&#39;/g, "'");
  }

  const tocItems = [];

  html = html.replace(
    /<h([2-3])(\s[^>]*)?>(.*?)<\/h[2-3]>/gs,
    (match, level, attrs, content) => {
      const plainText = unescape(content.replace(/<[^>]+>/g, ""));
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

  list.push({
    slug,
    title: data.title,
    description: data.description || "",
  });

  posts[slug] = {
    title: data.title,
    description: data.description || "",
    html,
    toc: tocItems,
  };
}

writeFileSync(listFile, JSON.stringify(list, null, 2));
writeFileSync(postsFile, JSON.stringify(posts, null, 2));
console.log(`✓ Learning data generated: ${list.length} posts → ${listFile} + ${postsFile}`);
