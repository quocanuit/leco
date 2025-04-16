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

1. **Initialize the environment**:
   ```bash
   make init
   ```

2. **Load and index data**:
   ```bash
   make index
   ```

3. **Start the server**:
   ```bash
   make up
   ```