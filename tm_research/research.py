from __future__ import annotations

from typing import Dict, Any, Optional, List

from .config import get_openai_llm, get_gemini_llm


def _topics_to_bullets(data: Dict[str, Any]) -> List[str]:
    bullets: List[str] = []
    for topic in data.get("topics", []):
        terms = ", ".join(t["term"] for t in topic.get("top_terms", [])[:8])
        bullets.append(f"Topic {topic['topic_id']}: {terms}")
    return bullets


def topics_to_markdown(data: Dict[str, Any], query: Optional[str] = None) -> str:
    lines: List[str] = []
    if query:
        lines.append(f"# Research summary for: {query}")
    else:
        lines.append("# Research summary")
    lines.append("")
    lines.append("## Topics")
    for topic in data.get("topics", []):
        lines.append(f"- Topic {topic['topic_id']}")
        terms = ", ".join(t["term"] for t in topic.get("top_terms", [])[:12])
        lines.append(f"  - Terms: {terms}")
        if topic.get("top_documents"):
            lines.append("  - Sources:")
            for doc in topic["top_documents"]:
                title = doc.get("title") or doc.get("url")
                lines.append(f"    - {title} ({doc.get('url')})")
    lines.append("")
    return "\n".join(lines)


def summarize_topics(
    data: Dict[str, Any],
    query: Optional[str] = None,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
) -> str:
    """Summarize topics with Gemini (preferred) or OpenAI if configured; otherwise fallback."""
    llm = None
    try:
        llm = get_gemini_llm(model=model, temperature=temperature)
    except ValueError:
        try:
            llm = get_openai_llm(model="gpt-4o-mini", temperature=temperature)
        except ValueError:
            return topics_to_markdown(data, query=query)

    from langchain_core.prompts import ChatPromptTemplate

    bullets = "\n".join(_topics_to_bullets(data))
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research assistant. Given topic modeling output and sources, write a concise, structured summary with key themes, insights, and links. Be factual and avoid speculation.",
            ),
            (
                "user",
                "Research question: {query}\n\nTopics (with key terms):\n{bullets}\n\nWrite a markdown summary with sections, bullets, and link the strongest sources per theme.",
            ),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"query": query or "(not provided)", "bullets": bullets})
    return result.content if hasattr(result, "content") else str(result)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


