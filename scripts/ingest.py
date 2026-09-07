#!/usr/bin/env python3
"""One-time ingestion script.

Run this once (and again any time the source document changes) before starting the API:

    python scripts/ingest.py

It chunks data/acme_corp_customer_policies.md, embeds the chunks locally, and writes a
FAISS index to vectorstore/. The API loads that index at startup; it does not re-ingest
on every request.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.vectorstore import build_vectorstore  # noqa: E402


def main() -> None:
    print(f"Ingesting: {settings.source_document}")
    store = build_vectorstore()
    count = store.index.ntotal
    print(f"Done. {count} chunks embedded and written to {settings.vectorstore_dir}")


if __name__ == "__main__":
    main()
