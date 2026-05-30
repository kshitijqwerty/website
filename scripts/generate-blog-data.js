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
      return hljs.highlightAuto(code).value;
    },
  })
);

marked.use({
  extensions: [
    {
      name: "mermaid",
      level: "block",
      start(src) {
        return src.indexOf("```mermaid\n");
      },
      tokenizer(src) {
        const match = /^```mermaid\n([\s\S]*?)\n```/.exec(src);
        if (match) {
          return {
            type: "mermaid",
            raw: match[0],
            text: match[1].trim(),
          };
        }
      },
      renderer(token) {
        return `<pre class="mermaid-pre"><div class="mermaid">\n${token.text}\n</div></pre>\n`;
      },
    },
  ],
});

marked.setOptions({
  breaks: true,
  gfm: true,
});

const __dirname = dirname(fileURLToPath(import.meta.url));
const contentDir = join(__dirname, "..", "content", "blog");
const outFile = join(__dirname, "..", "src", "data", "blogPosts.json");

const files = readdirSync(contentDir).filter((f) => f.endsWith(".md"));

const list = [];
const posts = {};

for (const file of files) {
  const raw = readFileSync(join(contentDir, file), "utf-8");
  const { data, content } = matter(raw);

  if (data.published === false) continue;

  const slug = data.slug || file.replace(/\.md$/, "");
  const html = marked(content)
    .replace(
      /<pre><code class="hljs language-(\w+)">/g,
      '<pre data-language="$1"><code class="hljs language-$1">'
    )
    .replace(
      /<pre><code>(?! )/g,
      '<pre data-language="txt"><code class="hljs">'
    );

  list.push({
    slug,
    title: data.title,
    date: data.date,
    description: data.description || "",
    tags: data.tags || [],
  });

  posts[slug] = {
    title: data.title,
    date: data.date,
    description: data.description || "",
    tags: data.tags || [],
    html,
  };
}

list.sort((a, b) => new Date(b.date) - new Date(a.date));

writeFileSync(outFile, JSON.stringify({ list, posts }, null, 2));
console.log(`✓ Generated blog data: ${list.length} posts → ${outFile}`);
