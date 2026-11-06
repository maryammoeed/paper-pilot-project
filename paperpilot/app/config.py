"""Runtime settings, read from environment variables or a local .env file (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """Load KEY=VALUE lines from .env. Real environment variables (e.g. hosting secrets) win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


load_dotenv()


@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "mock").lower()  # mock | groq | ollama | openai_compatible
    model: str = os.getenv("LLM_MODEL", "")
    base_url: str = os.getenv("LLM_BASE_URL", "")
    api_key: str = os.getenv("LLM_API_KEY", os.getenv("GROQ_API_KEY", ""))
    max_retries: int = int(os.getenv("EVAL_MAX_RETRIES", "3"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    mock_delay: float = float(os.getenv("MOCK_DELAY", "0.9"))  # seconds per agent call in demo mode
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1800"))
    # fair use for the host's own key (per visitor per hour, and per day for everyone)
    hosted_per_hour: int = int(os.getenv("HOSTED_REVIEWS_PER_HOUR", "3"))
    hosted_per_day: int = int(os.getenv("HOSTED_REVIEWS_PER_DAY", "25"))

    def resolved(self) -> tuple[str, str, str]:
        """Return (base_url, model, api_key) for the configured provider."""
        if self.provider == "groq":
            return (self.base_url or "https://api.groq.com/openai/v1",
                    self.model or "openai/gpt-oss-120b", self.api_key)
        if self.provider == "ollama":
            return (self.base_url or "http://localhost:11434/v1",
                    self.model or "llama3.1:8b", self.api_key or "ollama")
        return (self.base_url, self.model, self.api_key)


settings = Settings()
