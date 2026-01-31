# Slack Documentation Bot

A RAG-powered Slack bot that answers questions about your documentation and codebase using local embeddings and cloud LLM.

## Features

- **Slack Integration**: Responds to @mentions and direct messages
- **RAG Pipeline**: Indexes docs website + GitHub repo, retrieves relevant context
- **Source Citations**: Every answer includes links to source documents
- **Local Embeddings**: Uses HuggingFace bge-small-en-v1.5 for privacy
- **Socket Mode**: No public URL needed for development

## Quick Start

### 1. Clone and Install

```bash
cd ~/repos/slack-docs-bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:

- `SLACK_BOT_TOKEN`: Bot User OAuth Token (xoxb-...)
- `SLACK_APP_TOKEN`: App-Level Token (xapp-...) with `connections:write` scope
- `OPENAI_API_KEY`: API key for LLM responses
- `DOCS_SITEMAP_URL`: URL to your docs sitemap.xml
- `GITHUB_REPO_URL`: GitHub repo to index

### 3. Create Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Create new app → "From scratch"
3. Enable **Socket Mode** in Settings
4. Add **Event Subscriptions**:
   - `app_mention`
   - `message.im`
5. Add **Bot Token Scopes**:
   - `app_mentions:read`
   - `chat:write`
   - `im:history`
   - `im:read`
6. Install to workspace

### 4. Index Your Data

```bash
python scripts/reindex.py
```

### 5. Run the Bot

```bash
python -m src.bot.app
```

## Usage

- **@mention in channel**: `@your-bot how do I configure authentication?`
- **Direct message**: Just send any question

The bot will search the knowledge base and respond with an answer and source citations.

## Project Structure

```
slack-docs-bot/
├── src/
│   ├── config/          # Environment settings
│   ├── bot/             # Slack bot handlers
│   ├── ingestion/       # Document loaders
│   ├── storage/         # ChromaDB vector store
│   └── retrieval/       # RAG query engine
├── scripts/
│   └── reindex.py       # Manual re-indexing
├── tests/               # Unit tests
└── data/
    └── chroma_db/       # Vector store (gitignored)
```

## Troubleshooting

### Bot not responding

1. Check Socket Mode is enabled in Slack app settings
2. Verify `SLACK_APP_TOKEN` has `connections:write` scope
3. Check bot is invited to the channel

### Empty responses

1. Run `scripts/reindex.py` to populate the knowledge base
2. Verify data sources in `.env` are accessible

### LLM errors

1. Verify `OPENAI_API_KEY` is valid
2. Check API base URL if using custom endpoint

## License

MIT
