from __future__ import annotations

import json
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from .config import get_gemini_llm
from .ingestion import Record


def _build_docs_payload(records: List[Record], max_chars: int = 1200) -> str:
    lines: List[str] = []
    for i, r in enumerate(records):
        excerpt = (r.text[:max_chars] + "…") if len(r.text) > max_chars else r.text
        lines.append(f"- index: {i}\n  title: {r.title}\n  url: {r.url}\n  excerpt: {excerpt}")
    return "\n".join(lines)


def model_records_llm(
    records: List[Record],
    num_topics: int = 8,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Use an LLM (Gemini) to infer topics and map documents to topics.

    Returns a JSON-serializable dict matching the classic structure.
    """
    if not records:
        return {"num_documents": 0, "num_topics": 0, "topics": [], "documents": []}

    llm = get_gemini_llm(model=model, temperature=temperature)

    schema_example = {
        "num_documents": len(records),
        "num_topics": num_topics,
        "topics": [
            {
                "topic_id": 0,
                "label": "<short descriptive label>",
                "top_terms": ["term1", "term2", "term3"],
                "top_documents": [
                    {"index": 0, "url": "<url>", "title": "<title>", "score": 0.0}
                ],
            }
        ],
        "documents": [
            {
                "index": 0,
                "url": "<url>",
                "title": "<title>",
                "dominant_topic": 0,
                "score": 0.0,
            }
        ],
    }

    docs_block = _build_docs_payload(records)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a topic modeling assistant. Given a small corpus of web articles (index, title, url, excerpt), extract {num_topics} coherent, non-overlapping topics.\n"
                "Return STRICTLY valid JSON only, matching the provided schema.\n"
                "Use only the provided document indices and urls.\n"
                "Choose informative short labels and 8-12 top_terms per topic."
            ),
            (
                "user",
                "Schema example (illustrative, not the answer):\n{schema}\n\n"
                "Documents:\n{docs}\n\n"
                "Now produce the JSON answer using exactly {num_topics} topics.",
            ),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"num_topics": num_topics, "schema": json.dumps(schema_example, ensure_ascii=False, indent=2), "docs": docs_block})
    text = result.content if hasattr(result, "content") else str(result)

    # Try to load JSON directly; if model adds code fences, strip them
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`\n").split("\n", 1)[-1]
        if cleaned.startswith("json\n"):
            cleaned = cleaned[5:]
    data = json.loads(cleaned)

    # Ensure topic_id are ints and contiguous starting at 0
    topics = data.get("topics", [])
    for i, t in enumerate(topics):
        t["topic_id"] = int(t.get("topic_id", i))
        if "top_terms" in t and topics:
            # Convert list of dict terms to list[str] if needed
            if t["top_terms"] and isinstance(t["top_terms"][0], dict):
                t["top_terms"] = [x.get("term") for x in t["top_terms"] if isinstance(x, dict) and x.get("term")]

    # Ensure documents mapping exists
    if "documents" not in data or not isinstance(data["documents"], list):
        data["documents"] = []
        # Fallback: assign by first matching top_documents membership
        url_to_topic = {}
        for t in topics:
            for d in t.get("top_documents", []):
                url_to_topic[d.get("url")] = t["topic_id"]
        for i, r in enumerate(records):
            data["documents"].append(
                {
                    "index": i,
                    "url": r.url,
                    "title": r.title,
                    "dominant_topic": int(url_to_topic.get(r.url, 0)),
                    "score": 0.0,
                }
            )

    data["num_documents"] = len(records)
    data["num_topics"] = len(topics) if topics else num_topics
    return data


