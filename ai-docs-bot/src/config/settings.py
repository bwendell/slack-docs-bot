import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    SLACK_BOT_TOKEN: str
    SLACK_APP_TOKEN: str
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str
    LLM_MODEL: str
    DOCS_SITEMAP_URL: str
    GITHUB_REPO_URL: str
    CHROMA_PERSIST_DIR: str

def get_settings() -> Settings:
    return Settings(
        SLACK_BOT_TOKEN=os.getenv("SLACK_BOT_TOKEN", ""),
        SLACK_APP_TOKEN=os.getenv("SLACK_APP_TOKEN", ""),
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
        OPENAI_API_BASE=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        LLM_MODEL=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        DOCS_SITEMAP_URL=os.getenv("DOCS_SITEMAP_URL", ""),
        GITHUB_REPO_URL=os.getenv("GITHUB_REPO_URL", ""),
        CHROMA_PERSIST_DIR=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
    )
