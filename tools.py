# tools.py
import os
import json
from typing import List, Dict, Optional
from datetime import datetime

import streamlit as st
from openai import OpenAI

from utils import search_web


# ---------------- Keys / client ----------------
OPENAI_API_KEY: Optional[str] = (
    os.getenv("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_API_KEY")
)

if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI key missing. Set OPENAI_API_KEY in .env or in Streamlit Cloud Secrets.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------- Tool schema (lesson-style) ----------------
TOOLS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for fresh information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def invoke_model(messages: List[Dict], tools: Optional[List[Dict]] = None, model: str = "gpt-4o-mini") -> Dict:
    """
    Thin wrapper around Chat Completions (kept simple per lesson).
    Returns {"message": Message, "tool_calls": list}
    """
    completion = client.chat.completions.create(
        model=model,
        tools=tools,
        messages=messages,
    )
    msg = completion.choices[0].message
    return {"message": msg, "tool_calls": msg.tool_calls or []}


# ---------------- Summarizer (news -> concise text) ----------------
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_NEWS_PROMPT = (
    "Summarize for a retail investor in <= {max_chars} characters. "
    "Focus on catalysts, guidance, product/legal/macro impacts. "
    "Be concise and neutral.\n\n"
    "TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
)

def summarize_news_items(
    items: List[Dict],
    max_chars: int = 320,
    prompt_template: str = DEFAULT_NEWS_PROMPT,
    model: str = DEFAULT_MODEL,
) -> List[Dict]:
    """
    One chat call per item (simple & reliable).
    Output: [{"title","url","summary"} ...]
    """
    out: List[Dict] = []
    for it in items:
        title = it.get("title") or "Untitled"
        url = it.get("url") or ""
        content = it.get("content") or it.get("snippet") or ""
        prompt = prompt_template.format(max_chars=max_chars, title=title, url=url, content=content)

        messages = [
            {"role": "system", "content": "You are a helpful financial news summarizer."},
            {"role": "user", "content": prompt},
        ]
        res = client.chat.completions.create(model=model, messages=messages)
        summary = (res.choices[0].message.content or "").strip()
        if isinstance(max_chars, int) and max_chars > 0:
            summary = summary[:max_chars]
        out.append({"title": title, "url": url, "summary": summary})
    return out


# ---------------- Optional: example tool-calling pipeline (from your lesson) ----------------
def run_with_tool_calling(user_task: str, model: str = "gpt-4o-mini") -> str:
    """
    Demo of tool-calling per the lesson:
    - Ask model
    - If it requests web_search, call search_web()
    - Feed results back and get final answer
    """
    messages: List[Dict] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that can search the web. "
                f"Current date: {datetime.now():%Y-%m-%d}"
            ),
        },
        {"role": "user", "content": user_task},
    ]

    first = invoke_model(messages, tools=TOOLS, model=model)
    msg, calls = first["message"], first["tool_calls"]

    if calls:
        for call in calls:
            args = json.loads(call.function.arguments or "{}")
            if call.function.name == "web_search" and "query" in args:
                results = search_web(args["query"])
                messages.append(
                    {"role": "assistant", "content": f"Results for '{args['query']}':\n{results}"}
                )
        final = invoke_model(messages, tools=None, model=model)
        return final["message"].content or ""
    else:
        return msg.content or ""