import { readFileSync, writeFileSync, existsSync } from "fs";
import { execSync } from "child_process";
import path from "path";

const ROOT = process.cwd();
const TEX_PATH = path.join(ROOT, "content", "cv", "cv.tex");
const JSON_PATH = path.join(ROOT, "src", "data", "cv.json");
const PDF_LATEX_PATH = path.join(ROOT, "public", "kshitij_cv.pdf");


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



// ── Parse LaTeX → JSON ──
const tex = readFileSync(TEX_PATH, "utf-8");

const data = {
  _version: Date.now(),
  header: parseHeader(tex),
  summary: parseSummary(tex),
  skills: parseSkills(tex),
  experience: parseEntries(tex, "EXPERIENCE", "PROJECTS"),
  projects: parseEntries(tex, "PROJECTS", "EDUCATION"),
  education: parseEducation(tex),
};

writeFileSync(JSON_PATH, JSON.stringify(data, null, 2));
console.log("✓ Generated src/data/cv.json");

// ── LaTeX PDF ──
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
