# tools.py
import os
import streamlit as st
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

# ---- OpenAI (Responses API) ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # handled below

def _openai_client():
    if OpenAI is None:
        raise RuntimeError("Install openai: pip install openai")
    # Works with env or (optionally) Streamlit secrets
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI key not found. Set OPENAI_API_KEY in .env or Streamlit secrets.")
    return OpenAI(api_key=api_key)

DEFAULT_MODEL = "gpt-4o-mini"

# Prompt template with placeholders
DEFAULT_NEWS_PROMPT = (
    "Summarize the news item below for a retail investor. "
    "Max {max_chars} characters. Focus on what moved/ could move the stock, "
    "mention catalysts, guidance, products, legal or macro angles. "
    "Be concise and neutral. If the item is irrelevant to the ticker, say 'Skip'.\n\n"
    "TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
)

def summarize_news_items(
    items: List[dict],
    max_chars: int = 300,
    model: Optional[str] = None,
    prompt_template: str = DEFAULT_NEWS_PROMPT,
) -> List[Dict]:
    """
    Returns a list of dicts: [{"title", "url", "summary"}] in same order.
    """
    client = _openai_client()
    model = model or DEFAULT_MODEL
    out: List[Dict] = []
    for it in items:
        title = it.get("title") or "Untitled"
        url = it.get("url") or ""
        content = it.get("content") or it.get("snippet") or ""
        # Build concrete prompt
        prompt = prompt_template.format(title=title, url=url, content=content, max_chars=max_chars)

        # Responses API (preferred per OpenAI docs)
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
            )
            summary = getattr(resp, "output_text", None) or ""
        except Exception as e:
            summary = f"(summarization error: {e})"

        # Hard trim to max_chars (safety)
        if max_chars and isinstance(max_chars, int):
            summary = (summary[:max_chars]).rstrip()

        out.append({"title": title, "url": url, "summary": summary})
    return out