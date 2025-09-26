from __future__ import annotations

import json
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingestion import Record


def _top_terms_for_topic(
    topic_vector: np.ndarray, terms: np.ndarray, topn: int
) -> List[Dict[str, Any]]:
    order = np.argsort(topic_vector)[::-1][:topn]
    return [
        {"term": str(terms[idx]), "weight": float(topic_vector[idx])}
        for idx in order
    ]


def model_records(
    records: List[Record],
    num_topics: int = 8,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    max_df: float = 0.85,
    min_df: int = 2,
    topn_terms: int = 12,
) -> Dict[str, Any]:
    """Build a classic TF-IDF + NMF topic model from ingested records.

    Returns a JSON-serializable structure with topics, terms, and doc assignments.
    """
    texts = [r.text for r in records]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
    )
    tfidf = vectorizer.fit_transform(texts)

    nmf = NMF(
        n_components=num_topics,
        random_state=42,
        init="nndsvda",
        max_iter=500,
    )
    doc_topic = nmf.fit_transform(tfidf)
    topic_term = nmf.components_
    terms = vectorizer.get_feature_names_out()

    topics: List[Dict[str, Any]] = []
    for k in range(num_topics):
        top_terms = _top_terms_for_topic(topic_term[k], terms, topn_terms)
        # Find top documents for this topic by their weight
        doc_scores = doc_topic[:, k]
        top_doc_idx = np.argsort(doc_scores)[::-1][:5]
        top_documents = [
            {
                "index": int(i),
                "url": records[int(i)].url,
                "title": records[int(i)].title,
                "score": float(doc_scores[int(i)]),
            }
            for i in top_doc_idx
            if float(doc_scores[int(i)]) > 0.0
        ]
        topics.append(
            {
                "topic_id": k,
                "top_terms": top_terms,
                "top_documents": top_documents,
            }
        )

    # Dominant topic per document
    dominant = np.argmax(doc_topic, axis=1)
    dominant_scores = np.max(doc_topic, axis=1)
    documents = [
        {
            "index": i,
            "url": records[i].url,
            "title": records[i].title,
            "dominant_topic": int(dominant[i]),
            "score": float(dominant_scores[i]),
        }
        for i in range(len(records))
    ]

    return {
        "num_documents": len(records),
        "num_topics": num_topics,
        "topics": topics,
        "documents": documents,
    }


def save_topics_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_topics_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


