# RAG for legal consultation systems

## Introduction

LECO is an AI chatbot providing legal consultations, using a RAG system based on Vietnamese marriage legal judgments for accurate, contextual responses.

## Getting Started

### Prerequisites

- Python 3.x
- CUDA support (for PyTorch)
- Hugging Face API token
- Qdrant vector database server running

### Setup and Running

1. **Setting up runtime environment**
   
   Copy `.env` similar with `sample.env`, then run commands below
   ```bash
   export PYTHONPATH="$PYTHONPATH:$PWD"
   export USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
   ```

2. **Initialize the environment**:
   ```bash
   make init
   ```

3. **Load and index data**:
   ```bash
   make index
   ```

4. **Start the server**:
   ```bash
   make up
   ```

## Evaluation

```bash
cd eval
python simple_eval.py              # Quick evaluation
python ragas_simple.py             # RAGAS evaluation
```