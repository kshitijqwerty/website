export const projects = [
  {
    title: "AI Doc",

    desc: "AI-powered PDF summarizer that uses local LLMs to generate concise summaries, extract key insights, and enable intelligent document understanding without relying on external APIs.",

    tech: "Python, FastAPI, React, Ollama, Next.js",

    image: "/projects/ai_doc.png",

    demo: "https://ai-doc.kgup.me/",

    github: "https://github.com/kshitijqwerty/analyze-doc-local-llm",

    metrics: [
      "Local LLMs",
      "PDF summarization",
      "Insight extraction",
    ],

    tags: ["ML", "Backend"],

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

  // {
  //   title: "Video Hosting Platform",

  //   desc: "High-performance encrypted video hosting platform with adaptive streaming and scalable media pipelines.",

  //   tech: "Python, Django, FFmpeg, PostgreSQL, Docker, Kubernetes",

  //   image: "/projects/video-platform.png",

  //   demo: "https://video-demo.kgup.me/",

  //   github:
  //     "https://github.com/kshitijqwerty/video_hosting",

  //   metrics: [
  //     "99.9% uptime",
  //     "<120ms latency",
  //     "Auto-scaling",
  //   ],

  //   tags: ["Backend", "Infrastructure"],

  //   features: [
  //     "Encrypted HLS streaming",
  //     "Adaptive bitrate playback",
  //     "Video transcoding pipelines",
  //     "Async media processing",
  //   ],
  // },

  // {
  //   title: "ML Prediction Service",

  //   desc: "Scalable machine learning inference platform with autoscaling deployment pipelines and monitoring.",

  //   tech: "PyTorch, MLflow, AWS, FastAPI",

  //   image: "/projects/ml-service.png",

  //   demo: "https://ml-demo.kgup.me/",

  //   github:
  //     "https://github.com/kshitijqwerty/ml-service",

  //   metrics: [
  //     "~40ms inference",
  //     "A/B tested",
  //     "Autoscaling",
  //   ],

  //   tags: ["ML", "Backend"],

  //   features: [
  //     "Model registry integration",
  //     "Realtime inference APIs",
  //     "Experiment tracking",
  //     "Autoscaling inference workers",
  //   ],
  // },
];