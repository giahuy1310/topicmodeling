from __future__ import annotations

import os
from typing import Optional

import typer
from rich import print as rprint

from . import ingestion, modeling, research
from . import modeling_llm


app = typer.Typer(add_completion=False, help="Topic Modeling Research CLI")


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


@app.command()
def ingest(
    query: str = typer.Option(..., help="Web search query"),
    num_results: int = typer.Option(10, help="Number of search results to fetch"),
    out: str = typer.Option("data/raw.jsonl", help="Output JSONL file of documents"),
    delay: float = typer.Option(0.5, help="Delay between fetches (seconds)"),
):
    """Search the web and fetch cleaned page content."""
    _ensure_dir(out)
    rprint(f"[bold]Searching:[/bold] {query}")
    records = ingestion.ingest(query=query, num_results=num_results, delay_seconds=delay)
    ingestion.save_jsonl(out, records)
    rprint(f"Saved {len(records)} records to {out}")


@app.command()
def model(
    input: str = typer.Option("data/raw.jsonl", help="Input JSONL of documents"),
    out: str = typer.Option("data/topics.json", help="Output topics JSON"),
    num_topics: int = typer.Option(8, help="Number of topics"),
    method: str = typer.Option("llm", help="'llm' (Gemini) or 'classic' (TF-IDF+NMF)"),
    topn_terms: int = typer.Option(12, help="Top terms per topic"),
    max_features: int = typer.Option(5000, help="Max features for TF-IDF"),
    min_df: int = typer.Option(2, help="Min doc frequency"),
    max_df: float = typer.Option(0.85, help="Max doc frequency"),
):
    """Run TF-IDF + NMF topic modeling on ingested documents."""
    _ensure_dir(out)
    records = ingestion.load_jsonl(input)
    if method == "llm":
        data = modeling_llm.model_records_llm(records, num_topics=num_topics)
    else:
        data = modeling.model_records(
            records,
            num_topics=num_topics,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            topn_terms=topn_terms,
        )
    modeling.save_topics_json(out, data)
    rprint(f"Wrote topics to {out}")


@app.command()
def summarize(
    input: str = typer.Option("data/topics.json", help="Input topics JSON"),
    out: str = typer.Option("data/summary.md", help="Output markdown summary"),
    query: Optional[str] = typer.Option(None, help="Original research query (optional)"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model for summarization"),
    temperature: float = typer.Option(0.2, help="LLM temperature"),
):
    """Summarize topics into a readable research report. Uses LLM if configured."""
    _ensure_dir(out)
    data = modeling.load_topics_json(input)
    text = research.summarize_topics(data, query=query, model=model, temperature=temperature)
    research.write_text(out, text)
    rprint(f"Summary written to {out}")


@app.command()
def run(
    query: str = typer.Option(..., help="Web search query"),
    num_results: int = typer.Option(10, help="Number of search results to fetch"),
    num_topics: int = typer.Option(8, help="Number of topics"),
    method: str = typer.Option("llm", help="'llm' (Gemini) or 'classic' (TF-IDF+NMF)"),
    outdir: str = typer.Option("data", help="Output directory"),
):
    """End-to-end pipeline: ingest -> model -> summarize."""
    raw_path = os.path.join(outdir, "raw.jsonl")
    topics_path = os.path.join(outdir, "topics.json")
    summary_path = os.path.join(outdir, "summary.md")
    _ensure_dir(raw_path)
    _ensure_dir(topics_path)
    _ensure_dir(summary_path)

    rprint(f"[bold]Pipeline start[/bold]: {query}")
    records = ingestion.ingest(query=query, num_results=num_results, delay_seconds=0.5)
    ingestion.save_jsonl(raw_path, records)
    rprint(f"Ingested {len(records)} docs -> {raw_path}")

    if method == "llm":
        data = modeling_llm.model_records_llm(records, num_topics=num_topics)
    else:
        data = modeling.model_records(records, num_topics=num_topics)
    modeling.save_topics_json(topics_path, data)
    rprint(f"Modeled topics -> {topics_path}")

    text = research.summarize_topics(data, query=query)
    research.write_text(summary_path, text)
    rprint(f"Summary -> {summary_path}")


if __name__ == "__main__":
    app()


