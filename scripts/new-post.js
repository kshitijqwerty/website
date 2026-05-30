import { writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const contentDir = join(__dirname, "..", "content", "blog");

const title = process.argv[2];
if (!title) {
  console.error("Usage: npm run new-post \"My Blog Title\"");
  process.exit(1);
}

const slug = title
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, "-")
  .replace(/^-|-$/g, "");

const date = new Date().toLocaleDateString("en-US", {
  year: "numeric",
  month: "short",
});

const template = `---
title: "${title}"
date: "${date}"
slug: "${slug}"
description: ""
tags: []
published: true
---

Write your content here...
`;

mkdirSync(contentDir, { recursive: true });

const filePath = join(contentDir, `${slug}.md`);
writeFileSync(filePath, template);
console.log(`✓ Created ${filePath}`);
