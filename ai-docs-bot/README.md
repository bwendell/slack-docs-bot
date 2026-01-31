# ai-docs-bot

RAG-powered Slack bot for documentation Q&A.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system integrated with Slack, enabling users to ask questions about documentation and receive contextually relevant answers powered by LLMs.

## Features

- Document ingestion from web and GitHub sources
- Vector embeddings with HuggingFace models
- ChromaDB for semantic search
- **Flexible LLM backend**: OpenAI API or local Ollama
- Slack Bot Framework for message handling

## LLM Configuration

The bot supports two LLM providers, configurable via environment variables:

### Option 1: OpenAI API (Default)

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Option 2: Local Ollama (Offline/Privacy-Focused)

Run LLMs locally with [Ollama](https://ollama.com) for offline operation, privacy, or cost savings.

#### Quick Setup

**Option A: Using the setup script (Recommended)**

```bash
./scripts/ollama-setup.sh setup
```

This will install Ollama, start the server, and pull the default model.

**Option B: Manual setup**

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model (llama3.2 recommended for good balance of speed/quality)
ollama pull llama3.2

# 3. Configure environment
cat >> .env << EOF
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=120
OLLAMA_CONTEXT_WINDOW=8192
EOF

# 4. Start Ollama server (runs in background)
ollama serve &
```

**Option C: Docker Compose**

```bash
# Start Ollama in Docker
docker compose up -d ollama

# Pull a model into the container
docker compose exec ollama ollama pull llama3.2

# Configure to use Docker Ollama
echo "LLM_PROVIDER=ollama" >> .env
echo "OLLAMA_BASE_URL=http://localhost:11434" >> .env
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM backend: `openai` or `ollama` |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |
| `OLLAMA_CONTEXT_WINDOW` | `8192` | Context window size for the model |

#### Recommended Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `llama3.2` | 2GB | Fast | Good | Development, testing |
| `llama3.2:3b` | 2GB | Fast | Good | Default recommendation |
| `llama3.1:8b` | 4.7GB | Medium | Better | Production with good hardware |
| `mistral` | 4GB | Fast | Good | Alternative to Llama |
| `codellama` | 4GB | Medium | Best for code | Code-heavy documentation |

#### Verifying Ollama Setup

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list

# Test a model
ollama run llama3.2 "Hello, world!"
```

#### Troubleshooting Ollama

**Server not starting:**
```bash
# Check if already running
pgrep -f "ollama serve"

# Kill existing instance and restart
pkill -f "ollama serve"
ollama serve &
```

**Model not found:**
```bash
# Pull the model first
ollama pull llama3.2

# Verify it's available
ollama list
```

**Slow responses:**
- Ensure you have enough RAM (8GB+ recommended)
- Use a smaller model (e.g., `llama3.2` instead of `llama3.1:8b`)
- Check CPU/GPU utilization during inference

## Development

### Running Tests

```bash
# All tests (skips Ollama tests if not available)
pytest tests/ -v

# Only unit tests (no external dependencies)
pytest tests/test_settings.py tests/test_llm_provider.py -v

# E2E tests with Ollama (requires Ollama running with model pulled)
pytest tests/test_e2e_local.py -v --timeout=300
```

### Test Markers

- `@requires_ollama` - Test requires Ollama to be running with a model available
- Tests are automatically skipped if Ollama is not available

### Ollama Management Script

```bash
./scripts/ollama-setup.sh status   # Check Ollama status
./scripts/ollama-setup.sh start    # Start Ollama server
./scripts/ollama-setup.sh stop     # Stop Ollama server
./scripts/ollama-setup.sh pull     # Pull configured model
./scripts/ollama-setup.sh test     # Run a quick test query
```
