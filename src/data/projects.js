export const projects = [
  {
    title: "NeuralDocs AI",

    desc: "Self-hosted AI document intelligence platform — local LLM inference via Ollama with zero data egress. Deployable on VPS, Proxmox, or bare-metal via a single docker compose up.",

    tech: "Python, FastAPI, React, Ollama, Next.js",

    image: "/projects/ai_doc.png",

    demo: "https://ai-doc.kgup.me/",

    github: "https://github.com/kshitijqwerty/analyze-doc-local-llm",

    metrics: [
      "Local LLMs",
      "PDF summarization",
      "Insight extraction",
    ],

    tags: ["LLM", "RAG"],

    features: [
      "Upload and process PDFs",
      "AI-generated summaries",
      "Automatic keypoint extraction",
      "Local LLM inference with Ollama",
      "Privacy-focused document analysis",
      "Fast semantic document insights",
    ],
  },
  {
  title: "ONNXLab",

  desc: "Browser-native ONNX model visualization and inference platform. Upload any ONNX model, explore its graph interactively, and run real-time inference — entirely client-side with no backend required.",

  tech: "Next.js, React, TypeScript, TailwindCSS, ONNX Runtime Web, React Flow, Dagre",

  image: "/projects/onnxlab.png",

  demo: "https://onnxlab.kgup.me/",

  github: "https://github.com/kshitijqwerty/onnxlab",

  metrics: [
    "Client-side inference",
    "WebGPU accelerated",
    "Zero backend",
  ],

  tags: ["ML", "Frontend"],

  features: [
    "Interactive ONNX graph visualization with React Flow",
    "Netron-inspired node and tensor inspector",
    "Real-time inference via ONNX Runtime Web",
    "WebGPU acceleration with WASM fallback",
    "Auto-generated input UI from model metadata",
    "Image-to-tensor pipeline with dynamic resizing",
    "Top-K prediction viewer with custom label support",
    "Tensor analysis — shape, min, max, mean",
  ],
},

{
    title: "LLM Eval Harness",

    desc: "Automated evaluation pipeline for RAG and agentic LLM systems. Define test cases in YAML, run them across 4 model providers, and block bad deployments via a CI/CD hallucination gate.",

    tech: "Python, FastAPI, ChromaDB, PostgreSQL, GitHub Actions",

    image: "/projects/llm_eval.png",

    demo: null,

    github: "https://github.com/kshitijqwerty/llm-eval-harness",

    metrics: [
      "4 LLM providers",
      "CI/CD gate",
      "LLM-as-judge scoring",
    ],

    tags: ["LLM", "MLOps"],

    features: [
      "YAML-driven eval task definitions",
      "LLM-as-judge scoring (faithfulness, relevance, hallucination)",
      "Supports Ollama, Groq, OpenAI, and Claude",
      "Full RAG pipeline with ChromaDB + Sentence Transformers",
      "FastAPI dashboard with HTML report generation",
      "GitHub Actions CI/CD gate blocks hallucination regressions",
    ],
  },
  {
    title: "Image Similarity Search",

    desc: "Semantic image retrieval engine using CLIP embeddings and FAISS ANN indexing. Search millions of images by meaning, not metadata — with sub-millisecond latency.",

    tech: "Python, FastAPI, PyTorch, FAISS, CLIP, Streamlit",

    image: "/projects/image_search.png",

    demo: null,

    github: "https://github.com/kshitijqwerty/image-search-faiss",

    metrics: [
      "Sub-ms search latency",
      "CLIP ViT-B/32",
      "HNSW + IVF indexes",
    ],

    tags: ["ML", "CV"],

    features: [
      "CLIP ViT-B/32 semantic image embeddings",
      "FAISS HNSW, IVF, and exact index benchmarking",
      "Recall vs latency trade-off analysis",
      "Incremental indexing and embedding caching",
      "Metadata filtering on search results",
      "Streamlit frontend with top-K retrieval and similarity scores",
    ],
  }
];