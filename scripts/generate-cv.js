import { readFileSync, writeFileSync, existsSync } from "fs";
import { execSync } from "child_process";
import path from "path";
import puppeteer from "puppeteer";

const ROOT = process.cwd();
const TEX_PATH = path.join(ROOT, "content", "cv", "cv.tex");
const JSON_PATH = path.join(ROOT, "src", "data", "cv.json");
const PDF_LATEX_PATH = path.join(ROOT, "public", "kshitij_cv.pdf");
const PDF_WEB_PATH = path.join(ROOT, "public", "kshitij_cv_web.pdf");

function cleanText(text) {
  return text
    .replace(/\\href\{([^}]+)\}\{([^}]+)\}/g, "$2")
    .replace(/\\textbf\{([^}]+)\}/g, "$1")
    .replace(/\\textit\{([^}]+)\}/g, "$1")
    .replace(/\\texttt\{([^}]+)\}/g, "$1")
    .replace(/\\&/g, "&")
    .replace(/\\,/g, " ")
    .replace(/---/g, "—")
    .replace(/--/g, "–")
    .replace(/\\rightarrow/g, "→")
    .replace(/\\[a-zA-Z*]+(\[[^\]]*\])?/g, "")
    .replace(/[{}\\$]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function parseHeader(tex) {
  const m = tex.match(/\\begin\{center\}([\s\S]*?)\\end\{center\}/);
  if (!m) return { name: "", links: [], contact: "" };

  const center = m[1];
  const name = (center.match(/\\textbf\{([^}]+)\}/) || [])[1] || "";

  const linkRe = /\\href\{([^}]+)\}\{([^}]+)\}/g;
  const links = [];
  let match;
  while ((match = linkRe.exec(center)) !== null) {
    links.push({ url: match[1], label: match[2] });
  }

  let contact = center
    .replace(/\\href\{[^}]+\}\{[^}]+\}\s*\$\|\$\s*/g, "")
    .replace(/\\textbf\{[^}]+\}/, "")
    .replace(/\\normalsize|\\small|\\\[[\d.]+pt\]/g, "")
    .replace(/\$\|\$/g, "|")
    .replace(/[{}]/g, "")
    .replace(/\\/g, "")
    .replace(/\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return { name, links, contact };
}

function parseSummary(tex) {
  const m = tex.match(/\\section\*\{Summary\}([\s\S]*?)(?=\\section\*\{)/);
  if (!m) return "";
  return cleanText(m[1]);
}

function parseSkills(tex) {
  const m = tex.match(/\\begin\{tabbing\}([\s\S]*?)\\end\{tabbing\}/);
  if (!m) return [];

  return m[1]
    .split("\\\\")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const cat = (line.match(/\\textbf\{([^}]+)\}/) || [])[1];
      const tabIdx = line.search(/\\[=>]\s*/);
      const items =
        tabIdx >= 0
          ? cleanText(line.slice(tabIdx).replace(/\\[=>]\s*/, ""))
          : "";
      return cat ? { category: cat, items } : null;
    })
    .filter(Boolean);
}

function parseItemize(text) {
  const m = text.match(/\\begin\{itemize\}([\s\S]*?)\\end\{itemize\}/);
  if (!m) return [];
  return [...m[1].matchAll(/\\item\s+([\s\S]+?)(?=\\item|\\end\{itemize\}|$)/g)]
    .map((x) => cleanText(x[1]))
    .filter(Boolean);
}

function parseEntries(tex, sectionName, nextSection) {
  const pattern = nextSection
    ? `\\\\section\\*\\{${sectionName}\\}([\\s\\S]*?)(?=\\\\section\\*\\{${nextSection}\\})`
    : `\\\\section\\*\\{${sectionName}\\}([\\s\\S]*?)(?=\\\\section\\*\\{|\\\\end\\{document\\}|$)`;
  const sec = tex.match(new RegExp(pattern))?.[1] || "";
  const cleaned = sec.replace(/\\vspace\{[\d.]*pt\}/g, "");
  const blocks = cleaned.split(/\s*\\noindent\s*/).slice(1);

  if (sectionName === "EXPERIENCE") {
    return blocks.map((block) => {
      const bullets = parseItemize(block);
      const header = block.replace(/\\begin\{itemize\}[\s\S]*/, "").trim();
      const company =
        (header.match(/\\textbf\{([^}]+)\}/) || [])[1] || "";
      const title =
        (header.match(/\\textit\{([^}]+)\}/) || [])[1] || "";
      const afterHfill = header.replace(/.*\\hfill\s*/, "");
      const clean = afterHfill
        .replace(/\$\|\$/g, "|")
        .replace(/[{}]/g, "")
        .replace(/\\/g, "")
        .replace(/\s+/g, " ")
        .trim();
      const parts = clean.split("|").map((s) => s.trim());
      return {
        company,
        title,
        location: parts[0] || "",
        dates: parts.slice(1).join(" | ").trim(),
        bullets,
      };
    });
  }

  if (sectionName === "PROJECTS") {
    return blocks.map((block) => {
      const bullets = parseItemize(block);
      const header = block.replace(/\\begin\{itemize\}[\s\S]*/, "").trim();
      const name =
        (header.match(/\\textbf\{([^}]+)\}/) || [])[1] || "";
      const github =
        header.match(
          /\\href\{([^}]+)\}\{\\texttt\{[^}]*\}\}/,
        )?.[1] || "";
      const tech =
        (header.match(/\\textit\{([^}]+)\}/) || [])[1] || "";
      return { name, github, tech: cleanText(tech), bullets };
    });
  }

  return [];
}

function parseEducation(tex) {
  const sec =
    tex.match(
      /\\section\*\{EDUCATION\}([\s\S]*?)(?=\\section\*\{|\\end\{document\}|$)/,
    )?.[1] || "";
  const cleaned = sec.replace(/\\vspace\{[\d.]*pt\}/g, "");
  return cleaned
    .split(/\\noindent\s*/)
    .slice(1)
    .flatMap((block) =>
      block
        .split("\\\\")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const school =
            (line.match(/\\textbf\{([^}]+)\}/) || [])[1] || "";
          const afterSchool = line.replace(
            /\\textbf\{[^}]+\}\s*---\s*/,
            "",
          );
          const degree = afterSchool.replace(/\s*\\hfill.*/, "").trim();
          const dates = afterSchool.replace(/.*\\hfill\s*/, "").trim();
          return {
            school,
            degree: cleanText(degree),
            dates: cleanText(dates),
          };
        }),
    );
}

function generatePDFHtml(data) {
  const { header, summary, skills, experience, projects, education } = data;

  const linksHtml = header.links
    .map(
      (l, i) =>
        `${i > 0 ? '<span class="sep">/</span>' : ""}<a href="${l.url}">${l.label}</a>`,
    )
    .join("");

  const skillsHtml = skills
    .map(
      (s) => `
    <div class="skill-row">
      <span class="skill-cat">${s.category}</span>
      <span class="skill-val">${s.items}</span>
    </div>`,
    )
    .join("\n");

  const expHtml = experience
    .map(
      (e) => `
    <div class="exp-entry">
      <div class="exp-hdr">
        <div>
          <span class="exp-co">${e.company}</span>
          ${e.title ? `<span class="exp-title"> — ${e.title}</span>` : ""}
        </div>
        <span class="exp-meta">${e.location}${e.location && e.dates ? " — " : ""}${e.dates}</span>
      </div>
      ${e.bullets.length ? `<ul>${e.bullets.map((b) => `<li>${b}</li>`).join("\n")}</ul>` : ""}
    </div>`,
    )
    .join("\n");

  const projHtml = projects
    .map(
      (p) => `
    <div class="exp-entry">
      <div class="exp-hdr">
        <div>
          <span class="exp-co">${p.name}</span>
          ${p.github ? `<a href="${p.github}" class="gh-link">↗ GitHub</a>` : ""}
        </div>
        ${p.tech ? `<span class="exp-meta tech">${p.tech}</span>` : ""}
      </div>
      ${p.bullets.length ? `<ul>${p.bullets.map((b) => `<li>${b}</li>`).join("\n")}</ul>` : ""}
    </div>`,
    )
    .join("\n");

  const eduHtml = education
    .map(
      (e) =>
        `<div class="edu-row">
          <div><span class="edu-school">${e.school}</span>${e.degree ? `<span class="edu-degree"> — ${e.degree}</span>` : ""}</div>
          <span class="edu-dates">${e.dates}</span>
        </div>`,
    )
    .join("\n");

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  @page { size: A4; margin: 0.5in; }
  body {
    font: 9pt/1.35 'Helvetica Neue',Arial,sans-serif;
    color:#222; background:#fff; margin:0; padding:0;
  }
  .card {
    max-width:560pt; margin:0 auto;
    border:0.5pt solid #d4d4d4; border-radius:12pt;
    overflow:hidden;
  }
  .hd { text-align:center; padding:28pt 28pt 16pt; border-bottom:0.5pt solid #e5e5e5; }
  .hd h1 { font-size:20pt; font-weight:700; margin:0 0 6pt; letter-spacing:-0.01em; color:#111; }
  .hd .links { font-size:8.5pt; color:#555; margin-bottom:4pt; }
  .hd .links a { color:#555; text-decoration:none; }
  .hd .links a:hover { color:#4f46e5; }
  .hd .sep { color:#a3a3a3; margin:0 5pt; }
  .hd .contact { font-size:8.5pt; color:#737373; }
  .bd { padding:20pt 28pt; }
  .sec-hdr {
    display:flex; align-items:center; gap:6pt;
    font-size:9pt; font-weight:700; letter-spacing:0.05em; text-transform:uppercase;
    color:#171717; margin-bottom:12pt;
  }
  .sec-hdr .bar { display:inline-block; width:3pt; height:10pt; background:#4f46e5; border-radius:2pt; }
  .summary { color:#404040; margin-bottom:20pt; line-height:1.5; }
  .skills-grid { display:flex; flex-wrap:wrap; gap:4pt 20pt; margin-bottom:20pt; }
  .skill-row { display:flex; align-items:baseline; gap:4pt; width:220pt; }
  .skill-cat { font-weight:600; color:#111; white-space:nowrap; min-width:90pt; font-size:8.5pt; }
  .skill-val { color:#525252; font-size:8.5pt; }
  .exp-entry { border-left:1.5pt solid #c7d2fe; padding-left:10pt; margin-bottom:14pt; }
  .exp-hdr { display:flex; justify-content:space-between; align-items:baseline; gap:8pt; }
  .exp-co { font-weight:600; color:#111; }
  .exp-title { color:#525252; font-size:8.5pt; }
  .exp-meta { font-size:7.5pt; color:#737373; white-space:nowrap; }
  .gh-link { font-size:7.5pt; color:#4f46e5; text-decoration:none; margin-left:4pt; }
  .tech { white-space:normal; }
  ul { margin:4pt 0 0; padding-left:12pt; }
  li { color:#525252; margin-bottom:1.5pt; line-height:1.45; }
  .edu-row { display:flex; justify-content:space-between; align-items:baseline; gap:8pt; margin-bottom:4pt; }
  .edu-school { font-weight:600; color:#111; }
  .edu-degree { color:#525252; }
  .edu-dates { font-size:7.5pt; color:#737373; white-space:nowrap; }
</style></head><body>
  <div class="card">
    <div class="hd">
      <h1>${header.name}</h1>
      <div class="links">${linksHtml}</div>
      <div class="contact">${header.contact}</div>
    </div>
    <div class="bd">
      ${summary ? `<div class="summary">${summary}</div>` : ""}
      ${skills.length ? `<div class="sec-hdr"><span class="bar"></span>Skills</div><div class="skills-grid">${skillsHtml}</div>` : ""}
      ${experience.length ? `<div class="sec-hdr"><span class="bar"></span>Experience</div>${expHtml}` : ""}
      ${projects.length ? `<div class="sec-hdr"><span class="bar"></span>Projects</div>${projHtml}` : ""}
      ${education.length ? `<div class="sec-hdr"><span class="bar"></span>Education</div>${eduHtml}` : ""}
    </div>
  </div>
</body></html>`;
}

const tex = readFileSync(TEX_PATH, "utf-8");

const data = {
  header: parseHeader(tex),
  summary: parseSummary(tex),
  skills: parseSkills(tex),
  experience: parseEntries(tex, "EXPERIENCE", "PROJECTS"),
  projects: parseEntries(tex, "PROJECTS", "EDUCATION"),
  education: parseEducation(tex),
};

writeFileSync(JSON_PATH, JSON.stringify(data, null, 2));
console.log("✓ Generated src/data/cv.json");

(async () => {
  const html = generatePDFHtml(data);
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: "networkidle0" });
  await page.pdf({
    path: PDF_WEB_PATH,
    format: "A4",
    margin: { top: "0.5in", right: "0.5in", bottom: "0.5in", left: "0.5in" },
    printBackground: true,
  });
  await browser.close();
  console.log("✓ Generated public/kshitij_cv_web.pdf (Puppeteer)");
})().catch((err) => {
  console.error("Puppeteer PDF failed:", err.message);
});

function findPdflatex() {
  const candidates = [
    "/Library/TeX/texbin/pdflatex",
    "/usr/texbin/pdflatex",
  ];
  const glob = execSync(
    "ls /usr/local/texlive/*/bin/*/pdflatex 2>/dev/null || true",
    { encoding: "utf-8", shell: true },
  ).trim();
  if (glob) candidates.push(glob.split("\n")[0]);
  try {
    const which = execSync("which pdflatex 2>/dev/null", { encoding: "utf-8" }).trim();
    if (which) candidates.unshift(which);
  } catch {}
  return candidates.find((p) => existsSync(p)) || "";
}

const pdfLatexBin = findPdflatex();
if (pdfLatexBin) {
  try {
    const texDir = path.dirname(TEX_PATH);
    execSync(
      `${pdfLatexBin} -interaction=nonstopmode -output-directory=${texDir} ${TEX_PATH}`,
      { stdio: "pipe", cwd: texDir },
    );
    const cvPdf = path.join(texDir, "cv.pdf");
    if (existsSync(cvPdf)) {
      execSync(`mv "${cvPdf}" "${PDF_LATEX_PATH}"`);
      ["aux", "log", "out"].forEach((ext) => {
        const f = TEX_PATH.replace(/\.tex$/, `.${ext}`);
        if (existsSync(f)) execSync(`rm -f "${f}"`);
      });
      console.log("✓ Generated public/kshitij_cv.pdf (pdflatex)");
    }
  } catch {
    console.log("  (pdflatex error — skipping PDF)");
  }
} else {
  console.log("  (pdflatex not available — skipping PDF)");
}
