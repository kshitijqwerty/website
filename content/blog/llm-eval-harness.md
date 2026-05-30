---
title: "Building llm-eval-harness: A Production-Grade LLM Evaluation Pipeline with Free-Tier Tools"
date: "May 2026"
slug: "building-llm-eval-harness"
description: "How I went from zero LLM eval experience to shipping a production-grade RAG evaluation harness using only free-tier tools — and what every design decision taught me about building AI systems that can be trusted."
tags: [llm, evaluation, rag, mlops, python, fastapi, testing]
published: true
---

# LLMs Are Only as Good as Your Eval — Here's the Harness I Built to Measure Mine

## Overview

I built `llm-eval-harness` — a fully automated evaluation pipeline for RAG and agentic LLM systems. It runs test cases against multiple providers, scores responses with an LLM judge, tracks regressions over time, and blocks deployment when quality drops. It's the kind of tooling that would be at home in a real AI startup's internal infrastructure, and it cost me nothing to build — I used Ollama locally and Groq's free tier for everything.

This article covers the full architecture: the adapter pattern for multi-provider support, the ChromaDB-backed RAG pipeline, the LLM-as-judge scoring engine, Postgres persistence for regression detection, the FastAPI dashboard, and the GitHub Actions CI/CD gate. You'll learn what surprised me building this — from judge variance to YAML's unexpected scalability — and the tradeoffs I made along the way.

## Background

Everyone wants LLM evaluation experience. Nobody teaches it. Most LLM pipelines ship without any evaluation layer — when hallucinations slip through, there's no systematic way to catch regressions before they reach production.

I needed a way to answer a simple question with evidence: "Is this new model better than the last one?" Without eval infrastructure, you're guessing. With it, you have data. I wanted the kind of tooling that would be at home in a real AI startup's internal infrastructure — but built using only tools available for free.

## Goals

- **Build a complete RAG eval pipeline:** Ingest source documents into a vector store, retrieve relevant chunks per query, generate responses through multiple model providers, and score them automatically
- **LLM-as-judge scoring:** Evaluate faithfulness, relevance, and hallucination rate using an LLM judge independent of the model being tested
- **Regression detection across runs:** Compare each eval run against the previous one for the same task, catching quality degradation before it ships
- **CI/CD gate:** Exit with code 1 if hallucination regresses, blocking deployment — the non-negotiable requirement for production AI
- **Zero recurring cost:** Use only free-tier APIs and local models (Ollama, Groq free tier, ChromaDB, Postgres via Docker)

## Tech Stack

| Technology | Purpose | Reason for Selection |
|---|---|---|
| Ollama | Local model provider (llama3.2) | Free, runs entirely offline, no API key needed |
| Groq API | Judge model + cloud provider | Fast inference, generous free tier, llama3-8b-8192 as consistent judge |
| OpenAI / Anthropic | Optional premium providers | GPT-4o and Claude for comparison benchmarking |
| ChromaDB | Vector store for RAG | Zero-friction setup, `PersistentClient(path=".chroma")` just works |
| sentence-transformers (all-MiniLM-L6-v2) | Embedding model | Lightweight, runs locally, no API calls needed for embeddings |
| Postgres + SQLAlchemy | Persistence layer | Enables SQL queries for regression detection across runs |
| FastAPI + Uvicorn | REST API + dashboard | Async, auto-docs, Pydantic validation, simple background job tracking |
| Jinja2 | HTML report rendering | Self-contained reports with score bars and CI gate banners |
| GitHub Actions | CI/CD pipeline | Free for public repos, runs Postgres as a service container |

### Key Technology Decisions

**Ollama + Groq over all-cloud:** By supporting both local and cloud providers, the harness works in any environment. Development and debugging happens locally with zero latency. The same test cases run against Groq in CI without code changes.

**all-MiniLM-L6-v2 over OpenAI embeddings:** For RAG retrieval, you need an embedding model that runs locally, has predictable latency, and costs nothing. Sentence-transformers delivers all three. The quality difference vs. OpenAI's `text-embedding-3-small` is negligible for document chunk retrieval at this scale.

**LLM-as-judge over RAGAS or string metrics:** String metrics (BLEU, ROUGE) fail at semantic evaluation — "I don't know" and "The information is not available in the provided context" are semantically equivalent but share zero n-gram overlap. LLM-as-judge handles semantic equivalence naturally.

## Architecture / Design

### System Architecture

The pipeline flows through six stages: YAML task files → runner → provider adapters → LLM judge → Postgres → reports and CI gate.

```mermaid
flowchart TD
    A[YAML Task Files<br/>evals/tasks/*.yaml] --> B[Runner<br/>harness/runner.py]
    B --> C{Mode}
    C -->|direct| D[Adapter Layer<br/>harness/models.py]
    C -->|rag| E[ChromaDB<br/>Retrieve Top-K]
    E --> F[Augment Prompt<br/>Context + Query]
    F --> D
    D --> G1[OllamaAdapter]
    D --> G2[GroqAdapter]
    D --> G3[OpenAIAdapter]
    G1 --> H[LLM-as-Judge<br/>harness/metrics.py]
    G2 --> H
    G3 --> H
    H --> I[(Postgres<br/>EvalRun + EvalResultRow)]
    I --> J[Regression Check<br/>harness/regression.py]
    J --> K[HTML Report<br/>harness/reporter.py]
    J --> L[CI Gate<br/>exit code 1]
    J --> M[FastAPI Dashboard<br/>api/main.py]
```

### Data Flow

1. **Load tasks:** The runner reads YAML files from `evals/tasks/`. Each file contains eval cases — a query, an optional source document, the eval mode (`direct` or `rag`), and expected topics.
2. **Set up RAG context (if needed):** For `rag` mode, the runner calls `harness/rag.py` to ingest the source document into ChromaDB, chunk it, embed it with `all-MiniLM-L6-v2`, and build a retrieval pipeline.
3. **Run cases across providers:** The runner iterates over every combination of eval case × provider, calling the appropriate adapter and recording latency.
4. **Score with LLM-as-judge:** Each (query, response, source) triple is sent to Groq's `llama3-8b-8192` which returns faithfulness, relevance, and hallucination scores on a 0–1 scale.
5. **Persist to Postgres:** Each `EvalResult` is written as an `EvalResultRow` under an `EvalRun` record.
6. **Regression check:** Compare against the previous run for the same task. If any metric drops past the CI threshold, the CI gate triggers and `main.py` exits with code 1.
7. **Generate report:** `harness/reporter.py` renders a Jinja2 HTML template with score bars, per-model tables, and a CI gate banner.
8. **Expose through API:** A FastAPI dashboard lets you trigger runs, watch live logs, compare runs, and download reports.

### Key Design Decisions

**The adapter pattern.** `BaseLLMAdapter` defines a two-method contract (`complete()`, `model_name()`). Every provider implements it independently. The runner has zero provider-specific logic — adding `OpenAIAdapter` took very little time and zero changes to any other file.

**Judge model differs from evaluated model.** Using the same model to judge itself introduces self-serving bias. Groq's `llama3-8b-8192` serves as judge regardless of which model is being evaluated, creating a consistent external standard.

**Regression detection over absolute thresholds.** A faithfulness score that looks acceptable in isolation could hide trouble if it dropped from a previous run. Absolute scores tell you if you're above the floor; deltas tell you if you're moving in the right direction.

## Implementation Journey

### Initial Approach

I started with a single Python script that loaded a JSON file of test cases, called Ollama directly with `requests.post("http://localhost:11434/api/generate", ...)`, printed the responses, and I manually judged quality. Every "eval run" was me copy-pasting outputs into a Google Doc and scoring by hand. It worked as a proof of concept but had every problem a monolithic script can have — no provider abstraction, no persistence, no automation, no repeatable scoring.

### Challenges Faced

**LLM-as-judge is nondeterministic.** The same model response scored differently on different runs. I saw noticeable variance on a single case across repeated runs with identical inputs. LLM sampling is stochastic by default, and the judge prompt doesn't eliminate that.

**RAG evaluation is fundamentally harder than direct eval.** For direct mode, you send a prompt and score the answer. For RAG mode, you need to ingest documents, chunk them, embed them, retrieve relevant chunks, augment the prompt, *then* score the answer — and your scoring has to account for whether the model stayed faithful to what was retrieved, not just what was in the original document.

**Regression baselines break silently.** If you change a YAML task file (add a case, modify a query), the regression comparison becomes meaningless — you're comparing apples to oranges. The system couldn't detect this.

**The Postgres connection in CI was flaky.** GitHub Actions would start the Postgres service container, but `main.py` would try to connect before Postgres was ready, throwing a connection error. Without retry logic, the CI pipeline failed randomly.

### Solutions

**Accept judge variance and design around it.** The CI threshold exists partly because of judge variance — a threshold too tight would cause random failures unrelated to actual quality changes. A production system would run each case N times and use averages or confidence intervals. For a portfolio project, a coarse threshold with a documented tradeoff is the right call.

**Separate RAG and direct eval paths cleanly.** The runner has no hybrid mode and no magic inference about which to use. The task YAML specifies `mode: rag` or `mode: direct` explicitly. This makes eval results unambiguous and the code path simple to reason about.

**Document the regression baseline limitation.** The README and code explicitly note that changing task files breaks regression baselines silently. The solution — versioning task files by git commit hash — is on the roadmap but not yet implemented. Being honest about limitations is more credible than pretending they don't exist.

**Postgres health check with retry logic.** The GitHub Actions workflow uses `--health-cmd pg_isready` with `--health-interval 10s`, `--health-timeout 5s`, and `--health-retries 5`. Four lines of YAML that ensure Postgres is accepting connections before `python main.py` runs.

## Key Learnings

**LLM-as-judge is an approximation, not ground truth.** Judge variance means your CI threshold must be loose enough to tolerate noise but tight enough to catch real regressions. There's no perfect threshold — you calibrate empirically for your domain and judge model.

**YAML task files scale better than I expected.** I assumed YAML would feel limiting and I'd want Python test functions with proper assertions. That never happened. A handful of medical FAQ cases took about 15 minutes to write. The equivalent in pytest would have taken far longer and been harder to read. I had a non-programmer friend read `rag_support.yaml` who immediately proposed new test cases — that wouldn't have happened with Python test code.

**The adapter pattern pays for itself immediately.** By the time I was running cases against both Ollama and Groq simultaneously, the abstraction was clearly the right call. Adding `OpenAIAdapter` took very little time and zero changes to any other file.

**Regression detection changes how you think about evaluation.** Before building it, I thought about evaluation as "does this run pass?" After building it, I started thinking about evaluation as "is this run *better than* the last run?" That's a fundamentally different mental model, and it's the right one for systems that evolve over time.

**ChromaDB genuinely has zero friction.** I expected setting up a vector store to be complicated. `chromadb.PersistentClient(path=".chroma")` and you're done. The persistence just works, the API is clean, and for portfolio-scale RAG it's the right tool.

## Results

The harness runs several eval suites covering different domains across two providers (Ollama and Groq), with OpenAI and Anthropic as drop-in additions. A full eval run takes a few minutes. The CI gate catches regressions and blocks deployment with an HTML report showing exactly which metric dropped and by how much. The dashboard is browser-accessible with live job logs, run history, and one-click re-triggering.

The judge variance measurements informed the CI threshold — conservative enough to avoid false positives while catching genuine regressions.

![HTML Eval Report — sample_rag_eval run showing score bars for faithfulness, relevance, and hallucination across both providers](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/qlcnnt74lq3477gyf5ut.png)

## What I Would Do Differently

**Start with the adapter pattern from day one.** I built the Ollama adapter, hacked in Groq support with `if/else` branches, then refactored to the adapter pattern. The upfront abstraction would have saved an hour of cleanup.

**Add task file versioning before the first run.** Changing a YAML task file breaks regression baselines silently. Versioning task files by git commit hash would make baselines unambiguous. This is the single biggest quality-of-life improvement I'd make.

**Use a persistent job queue instead of threading.** The in-memory job tracker works for demo purposes but is the most fragile part of the system. Server restart = lost jobs. A Postgres-backed job table with proper state transitions would make the dashboard production-ready.

**Implement multi-run averaging from the start.** Running each case N times and reporting mean ± std would make CI thresholds much more reliable. The judge variance I documented would be visible in the confidence intervals instead of silently inflating false-positive risk.

## Future Roadmap

- **Task file versioning by git commit hash** — Link eval runs to specific task file versions so regression baselines survive task edits
- **Persistent job queue** — Replace the in-memory dict with a Postgres-backed job table with state transitions and retry logic
- **Async eval runner** — Use `asyncio` and `httpx` for concurrent provider calls, cutting total run time by roughly (number of providers) × (average latency per case)
- **Custom judge prompts per domain** — Medical FAQ evaluation should weight hallucination more heavily. Let task authors define judge configuration in their YAML files
- **Dashboard authentication** — No auth currently; fine for local use, necessary for shared deployments
- **Multi-run averaging** — Run each case N times and report mean ± std to make CI thresholds reliable despite judge variance
- **Streaming responses** — Add streaming support to adapters for faster perceived latency and partial scoring during generation

## Code Examples

**The adapter pattern — every provider implements the same interface:**

```python
from abc import ABC, abstractmethod

class BaseLLMAdapter(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...

    @abstractmethod
    def model_name(self) -> str:
        ...

class GroqAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def model_name(self) -> str:
        return f"groq/{self.model}"
```

**LLM-as-judge prompt — structured for consistent JSON output:**

```python
JUDGE_PROMPT = """
You are an expert evaluator for language model outputs.

Given:
- Query: {query}
- Source Document: {context}
- Model Response: {response}

Score the response on each dimension from 0.0 to 1.0:

1. FAITHFULNESS: Does the response accurately reflect the source document?
   (1.0 = fully grounded, 0.0 = contradicts the source)

2. RELEVANCE: Does the response actually answer the query?
   (1.0 = directly and completely answers, 0.0 = completely off-topic)

3. HALLUCINATION: Does the response introduce facts not in the source?
   (1.0 = no hallucination, 0.0 = heavily hallucinated)

Respond with JSON only:
{{"faithfulness": float, "relevance": float, "hallucination": float}}
"""
```

**Regression detection — compute deltas against the previous run:**

```python
@dataclass
class MetricDiff:
    metric: str
    previous: float
    current: float
    delta: float

@dataclass
class RegressionReport:
    task_id: str
    provider: str
    diffs: list[MetricDiff]
    has_regression: bool
    regression_metrics: list[str]

def compute_diff(task_id: str, provider: str, db: Session) -> RegressionReport:
    runs = (
        db.query(EvalRun)
        .filter(EvalRun.task_file.contains(task_id))
        .order_by(EvalRun.started_at.desc())
        .limit(2)
        .all()
    )
    if len(runs) < 2:
        return RegressionReport(task_id, provider, [], False, [])

    current = runs[0].average_scores(provider)
    previous = runs[1].average_scores(provider)
    diffs = [
        MetricDiff(m, prev, curr, curr - prev)
        for (m, prev, curr) in zip(SCORE_METRICS, previous, current)
    ]
    regression_metrics = [d.metric for d in diffs if d.delta < -CI_THRESHOLD]
    return RegressionReport(
        task_id, provider, diffs,
        has_regression=len(regression_metrics) > 0,
        regression_metrics=regression_metrics,
    )
```

**CI/CD workflow — the GitHub Actions gate:**

```yaml
name: LLM Eval CI
on: [push, pull_request]

services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: eval_db
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5

env:
  DATABASE_URL: postgresql://postgres:postgres@localhost:5432/eval_db
  GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}

steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with:
      python-version: '3.11'
  - run: pip install -r requirements.txt
  - run: python main.py
  - uses: actions/upload-artifact@v4
    if: always()
    with:
      name: eval-report-${{ github.sha }}
      path: reports/
```

**Eval task YAML format:**

```yaml
- id: refund_window
  mode: rag
  query: "What is the refund window for purchases?"
  document: docs/support_policy.txt
  expected_topics:
    - "30 days"
    - "original payment method"
  tags:
    - policy
    - refunds
```

## Key Takeaways

1. **LLM-as-judge is nondeterministic.** Design your CI threshold around ±0.05–0.08 variance. Multi-run averaging is the correct long-term fix, but a coarse threshold with documented limitations works for a portfolio project.
2. **Adapters are the single most important abstraction in a multi-provider system.** A two-method interface eliminates provider-specific logic from every other component. It paid for itself immediately when I added the second provider.
3. **YAML task files are more accessible and faster to author than test code.** Non-engineers can read and contribute eval cases. A handful of cases take minutes to write. The format scales surprisingly well.
4. **Regression detection changes your mental model.** Absolute scores tell you if you're above the floor. Deltas tell you if you're moving in the right direction. Both are necessary for trustworthy evaluation.
5. **The judge model must differ from the evaluated model.** Self-judging introduces self-serving bias. A consistent external judge — even a smaller 8B model — is more reliable than self-evaluation.
6. **Always upload CI artifacts on failure.** `if: always()` on report uploads means you can debug regressions without running blind. A failed run with no artifact is a debugging dead end.
7. **ChromaDB has zero friction for portfolio-scale RAG.** `PersistentClient(path=".chroma")` is all you need. No Docker, no configuration, no ops.
8. **Postgres health checks in CI are mandatory.** A container that isn't accepting connections yet is indistinguishable from a downed database. Add `pg_isready` retry logic from day one.
9. **Document your tradeoffs explicitly.** In-memory job store, single judge model, fixed CI threshold — each should have a comment explaining the limitation. Honesty is more credible than pretending limitations don't exist.
10. **Free-tier tools are sufficient for production-grade infrastructure.** Ollama, Groq, ChromaDB, Postgres via Docker, GitHub Actions — all free. The constraint forces intentional design decisions that make the system better.

## Conclusion

I built a complete LLM evaluation pipeline that answers "is the new model better?" with evidence instead of intuition. The harness runs test cases across multiple providers, scores responses with an independent judge, tracks regressions across runs, and blocks deployment when quality drops — all using free-tier tools.

Building this project taught me more about production AI systems than any course I've taken. Not because the code is complex — it isn't — but because the problems you have to think through (what do you measure? how do you measure it? what counts as a regression? how do you prevent false positives in CI?) are the same problems real teams are solving every day.

LLM evaluation is infrastructure work. It's not glamorous, it doesn't produce a flashy demo, and it requires you to think carefully about what "good" means for a given task. But it's exactly the kind of work that separates teams shipping reliable AI products from teams shipping demos.

If you're trying to build LLM eval experience without access to a company's internal infrastructure, I hope this project gives you a concrete starting point. Fork it, write your own eval cases, and start measuring.

**GitHub:** [github.com/kshitijqwerty/llm-eval-harness](https://github.com/kshitijqwerty/llm-eval-harness)
