# Gemma 4 Multimodal Reasoner

A production-ready multimodal reasoning engine powered by [Gemma 4](https://ai.google.dev/gemma/docs/core).
Analyze documents, charts, math problems, screenshots, and more — with chain-of-thought
reasoning, configurable thinking mode, and support for all four Gemma 4 model sizes.

## Features

- **Multimodal reasoning** — text + image input with interleaved modality support
- **Thinking mode** — chain-of-thought reasoning via Gemma 4's native `<|think|>` control token
- **Document analysis** — charts, tables, receipts, forms, diagrams, handwriting
- **Math solving** — visual math problems with step-by-step reasoning
- **Screen understanding** — UI analysis for agentic workflows
- **Multi-image comparison** — compare and contrast multiple images
- **Multi-turn conversation** — contextual chat with image context
- **Multi-backend** — Ollama, HuggingFace Transformers, or OpenAI-compatible APIs
- **All model sizes** — E2B (edge) through 31B (workstation)

## Quick Start

```bash
# Install
pip install "gemma4-reasoner[ollama]"

# Pull a model
ollama pull gemma4:e4b

# Analyze an image
gemma4 analyze --image chart.png --doc-type chart

# Ask a question
gemma4 analyze --image photo.jpg --question "What's happening here?"

# Compare two images
gemma4 compare --images before.png after.png --question "What changed?"

# Solve a math problem
gemma4 math --image problem.jpg

# Interactive chat
gemma4 chat --image screenshot.png
```

## Model Selection

| Model | Size | Context | Modalities | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **E2B** | 2.3B effective | 128K | Text, Image, Audio | Mobile, IoT, real-time |
| **E4B** | 4.5B effective | 128K | Text, Image, Audio | Laptops, edge servers |
| **26B A4B** | 3.8B active | 256K | Text, Image | Fast inference, coding |
| **31B** | 30.7B dense | 256K | Text, Image | Max quality, reasoning |

## Architecture

```text
┌──────────────────────────────────────────────┐
│                 CLI / API                     │
├──────────────────────────────────────────────┤
│             Gemma4Reasoner                    │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐  │
│  │Thinking  │ │  Image    │ │  Prompt    │  │
│  │  Mode    │ │ Processor │ │  Builder   │  │
│  └──────────┘ └───────────┘ └────────────┘  │
├──────────────────────────────────────────────┤
│          Backend Adapter Layer                │
│  ┌────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Ollama │ │HuggingFace│ │ OpenAI-Compat │  │
│  └────────┘ └──────────┘ └───────────────┘  │
├──────────────────────────────────────────────┤
│          Gemma 4 Model (E2B → 31B)           │
└──────────────────────────────────────────────┘
```

## Python API

```python
from gemma4_reasoner import Gemma4Reasoner, ReasoningConfig, ModelSize

config = ReasoningConfig(
    model_size=ModelSize.E4B,
    backend="ollama",
    thinking=True,
)
reasoner = Gemma4Reasoner(config)

# Analyze a chart
result = reasoner.analyze_document("sales_chart.png", doc_type="chart")
print(result.response)

# Solve a math problem
result = reasoner.solve_math_visual("problem.jpg")
print(result.response)
print(result.thinking)  # Chain-of-thought reasoning

# Compare images
result = reasoner.compare_images(
    ["before.png", "after.png"],
    question="What changed?"
)

# Multi-turn chat
r1 = reasoner.chat("What's in this image?", images=["photo.jpg"])
r2 = reasoner.chat("Tell me more about the main subject")
```

## Docker

```bash
docker compose up --build
gemma4 analyze --image /data/chart.png --doc-type chart
```

## License

Apache 2.0
