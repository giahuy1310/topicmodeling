from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from a local .env file if present
load_dotenv()


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def get_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Return a LangChain ChatOpenAI instance if configured.

    Raises a ValueError if OPENAI_API_KEY is missing.
    """
    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Provide it in your environment or .env to use LLM features."
        )

    # Lazy import to keep base install lightweight if LLMs aren't used
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def get_gemini_llm(model: str = "gemini-1.5-flash", temperature: float = 0.2):
    """Return a LangChain ChatGoogleGenerativeAI instance if configured.

    Raises a ValueError if GOOGLE_API_KEY is missing.
    """
    api_key = get_env("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Provide it in your environment or .env to use Gemini features."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

