# Topic Modeling Research (LangChain)

LangChain-based toolkit for researching topics from web sources. Includes

-
- Ingestion: web search + content fetching and cleaning
- Topic modeling: TF-IDF + NMF to extract topics and keywords
- LLM research (optional): summarize and refine topics with an LLM
- CLI: run end-to-end pipeline or individual steps

## Quickstart

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Configure environment

Create a `.env` file (or copy `env.example` to `.env`) and set your API keys as needed.

1. Run a sample research pipeline (LLM default: Gemini)

```bash
python -m tm_research.cli run --query "quantum computing benchmarks" --num-results 10 --num-topics 8
```

## Commands

```bash
python -m tm_research.cli ingest --query "<your query>" --num-results 10 --out data/raw.jsonl
python -m tm_research.cli model --input data/raw.jsonl --num-topics 8 --out data/topics.json
python -m tm_research.cli summarize --input data/topics.json --out data/summary.md
python -m tm_research.cli run --query "<your query>" --num-results 10 --num-topics 8
```

To use classic TF-IDF+NMF instead of LLM:

```bash
python -m tm_research.cli model --input data/raw.jsonl --num-topics 8 --method classic --out data/topics.json
python -m tm_research.cli run --query "<your query>" --num-results 10 --num-topics 8 --method classic
```

## Project structure

```text
tm_research/
  __init__.py
  config.py
  ingestion.py
  modeling.py
  research.py
  cli.py
data/
  (created at runtime)
env.example
requirements.txt
```

## Notes

- LLM features are optional. If you set `OPENAI_API_KEY` or compatible provider keys in `.env`, summarization will use an LLM via LangChain.
- Topic modeling uses classic TF-IDF + NMF for reproducibility and speed. Swap for BERTopic or LDA as desired.

### Gemini setup

- Set `GOOGLE_API_KEY` in `.env` to enable Gemini via LangChain.
- The CLI defaults to Gemini-based topic modeling; use `--method classic` for TF-IDF+NMF.
