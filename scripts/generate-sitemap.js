import { readFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const blogListFile = join(__dirname, "..", "src", "data", "blogList.json");
const learningListFile = join(__dirname, "..", "src", "data", "learningList.json");
const outFile = join(__dirname, "..", "public", "sitemap.xml");

const SITE = "https://kgup.me";
const today = new Date().toISOString().split("T")[0];

const staticPages = [
  { loc: "/", priority: "1.0", changefreq: "monthly" },
  { loc: "/blog", priority: "0.8", changefreq: "weekly" },
  { loc: "/learning", priority: "0.8", changefreq: "weekly" },
];

let blogPages = [];
try {
  const list = JSON.parse(readFileSync(blogListFile, "utf-8"));
  blogPages = list.map((p) => {
    const d = new Date(p.date);
    const lastmod = isNaN(d.getTime()) ? today : d.toISOString().split("T")[0];
    return {
      loc: `/blog/${p.slug}`,
      priority: "0.6",
      changefreq: "monthly",
      lastmod,
    };
  });
} catch {
  blogPages = [];
}

let learningPages = [];
try {
  const list = JSON.parse(readFileSync(learningListFile, "utf-8"));
  learningPages = list.map((p) => ({
    loc: `/learning/${p.slug}`,
    priority: "0.6",
    changefreq: "monthly",
    lastmod: today,
  }));
} catch {
  learningPages = [];
}

const all = [
  ...staticPages,
  ...blogPages,
  ...learningPages,
];

const urls = all
  .map(
    (p) => `  <url>
    <loc>${SITE}${p.loc}</loc>
    <lastmod>${p.lastmod || today}</lastmod>
    <changefreq>${p.changefreq}</changefreq>
    <priority>${p.priority}</priority>
  </url>`
  )
  .join("\n");

const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>
`;

writeFileSync(outFile, xml);
console.log(`✓ Sitemap generated: ${outFile} (${all.length} URLs)`);
