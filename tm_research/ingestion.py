from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


@dataclass
class Record:
    url: str
    title: str
    text: str


def search_web(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query, max_results=num_results)):
            if not r:
                continue
            results.append({
                "title": r.get("title") or "",
                "href": r.get("href") or r.get("link") or "",
                "body": r.get("body") or "",
            })
            if i + 1 >= num_results:
                break
    return results


def fetch_and_clean(url: str, timeout: int = 15) -> Record:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Prefer main/article; fallback to body
    main = soup.find("main") or soup.find("article") or soup
    # Remove scripts/styles/navs
    for tag in main.find_all(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        tag.decompose()
    title = (soup.title.string if soup.title and soup.title.string else url).strip()
    text = "\n".join(p.get_text(" ", strip=True) for p in main.find_all(["p", "li"]))
    text = " ".join(text.split())
    return Record(url=url, title=title, text=text)


def ingest(query: str, num_results: int = 10, delay_seconds: float = 0.5) -> List[Record]:
    search_results = search_web(query, num_results)
    records: List[Record] = []
    for r in search_results:
        url = r.get("href")
        if not url:
            continue
        try:
            rec = fetch_and_clean(url)
            if rec.text:
                records.append(rec)
        except Exception:
            # Skip failures; keep the pipeline robust
            pass
        time.sleep(delay_seconds)
    return records


def save_jsonl(path: str, records: Iterable[Record]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Record]:
    out: List[Record] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(Record(**obj))
    return out

