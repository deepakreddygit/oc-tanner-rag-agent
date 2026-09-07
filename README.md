# Acme Corp Policy Agent

RAG agent built for the O.C. Tanner AI Engineer take-home. It reads a customer policy
document and answers questions about it through a single `POST /chat` endpoint —
grounded in the document, not the model's own knowledge, and it says so plainly when
the document doesn't have the answer.

**Live demo:** https://oc-tanner-rag-agent.onrender.com/docs — try `POST /chat` right
there in the browser. Heads up: it's on Render's free tier, so if nobody's hit it in a
while the first request takes 30-60 seconds to wake back up. That's normal, not a bug.

## What it does

- One-time ingestion step chunks `data/acme_corp_customer_policies.md`, embeds the
  chunks, and stores them in a FAISS index.
- A LangGraph agent retrieves the relevant chunks for a question — using hybrid search
  (keyword + dense, more on that below) — and answers using only those, summarizing in
  its own words when asked to summarize, and refusing when the document doesn't cover
  something instead of guessing.
- FastAPI wraps that as `POST /chat`, with per-session chat history so follow-ups work
  ("what about damaged ones?").

The sample document wasn't actually attached to the assignment, so I wrote one myself
to match the example given in the brief (30-day refund window, 10% restocking fee on
electronics over $500). It covers returns, shipping, cancellations, damaged items,
warranty, membership, price adjustments, and support hours. It deliberately doesn't say
anything about shipping outside the US/Canada — that's what the "can't answer this"
example below is testing.

## How it works

```
user message -> contextualize -> retrieve -> generate -> answer
```

Three nodes, no loops, no conditional branches:

1. **contextualize** — turns a follow-up question into a standalone one using the
   session's chat history (e.g. "is there a restocking fee on that?" becomes "...on the
   laptop return discussed above?"). Skipped entirely on the first message of a session
   since there's no history to work from yet.
2. **retrieve** — hybrid search: BM25 (keyword/lexical match against the FAISS index's
   underlying chunks) and dense vector search (the FAISS index from `scripts/ingest.py`)
   run in parallel, and their two ranked lists get merged by reciprocal rank fusion into
   one. BM25 catches exact terms a dense embedding can blur together (a specific fee
   percentage, a product name); dense search catches paraphrases BM25 would miss
   entirely ("money back" for "refund"). Top-k after fusion goes to the prompt.
3. **generate** — answers strictly from those chunks. The prompt tells it to say it
   can't answer from the available documents rather than guess, and to treat retrieved
   text as reference material only, never as instructions to follow (a basic guard in
   case someone poisons the source document).

I thought about adding a separate "check if the answer is actually grounded" node that
loops back and regenerates on a bad score — a lot of RAG setups do this. For one small,
static document it felt like a second LLM call to catch something the generation prompt
already mostly handles on its own, so I left it out. I'd reconsider for a bigger, noisier
corpus where retrieval quality actually varies a lot.

`scripts/llm.py` and `scripts/vectorstore.py` are the only two files that know which
provider they're talking to (OpenAI, HuggingFace/FAISS) or how retrieval is actually
implemented (BM25 + FAISS fused). The agent itself depends only on LangChain's generic
`BaseChatModel`/`BaseRetriever` interfaces — it calls `retriever.invoke(query)` and has
no idea two retrievers are involved under the hood, which is also what makes it easy to
swap in fakes for testing.

## Layout

Kept flat on purpose: `app.py` at the root is the one thing you run — it wires
everything together into the FastAPI service and starts it. Everything it depends on
(the agent graph, config, the vector store, retries, rate limiting, etc.) lives in
`scripts/`, alongside the one-time ingestion script. There's no separate installable
`app` package here; it isn't needed for a single-service project this size.

## Retries and rate limiting

This is actually deployed and public now, not just running on my laptop, so a couple of
things needed handling that wouldn't matter for a local script:

- LLM calls retry up to 3 times with backoff, but only for errors where retrying
  actually helps — rate limits, timeouts, 5xxs. Auth errors and bad input fail
  immediately instead, since retrying those would just waste time.
- `/chat` is capped at 20 requests/minute per IP so nobody can accidentally (or on
  purpose) run up the OpenAI bill. The IP is read from `X-Forwarded-For` since Render
  puts the app behind a proxy.

Both live in memory in the same process — fine for a single free-tier instance, not
something I'd ship as-is if this ran on more than one.

## Setup, from zero

You'll need Python 3.10+ and an OpenAI API key.

1. Get a key at platform.openai.com/api-keys (add a few dollars of credit first —
   running this end to end costs a few cents, not more).
2. Clone and install:
   ```bash
   git clone <this-repo-url>
   cd oc-tanner-rag-agent
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt -r requirements-local-embeddings.txt
   ```
   The second requirements file pulls in the local HuggingFace embedding model used by
   default. It's kept separate from `requirements.txt` on purpose — see Deployment below
   for why the deployed version skips it.
3. `cp .env.example .env`, then open it and paste your key in as `OPENAI_API_KEY=sk-...`.
   Everything else in there already has a sensible default.
4. `python scripts/ingest.py` — chunks the document, embeds it locally, writes the FAISS
   index to `vectorstore/`. Only needs to run once, or again if the source doc changes.
5. `python app.py` — serves on `http://127.0.0.1:8000`, with interactive docs at
   `/docs`. (If you're actively editing and want auto-reload, `uvicorn app:app --reload`
   does the same thing.)

## Trying it

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "message": "What is the refund window, and is there a restocking fee?"}'
```

A full transcript covering all three required behaviors (a factual question, a summary
request, and a question the document can't answer) is in [`sample_io.md`](sample_io.md).

## Deployment

Live on [Render](https://render.com)'s free tier at
`https://oc-tanner-rag-agent.onrender.com`. Two things differ from local dev:

Embeddings come from OpenAI's API instead of the local HuggingFace model
(`EMBEDDING_PROVIDER=openai`), and this isn't just a nice-to-have. The local embedding
model depends on `torch`, and torch's default Linux install pulls in a full CUDA
toolkit regardless of whether there's a GPU around — that alone blew past the free
tier's 512MB limit during the build, before the app even got a chance to start.
Switching to OpenAI's embeddings API for the deployed instance drops that dependency
entirely. Local dev keeps using the free local model by default.

Ingestion also runs during the build (`pip install -r requirements.txt && python
scripts/ingest.py`), so the FAISS index already exists by the time the server starts.
`render.yaml` has this wired up if you use Render's "Blueprint" import — a plain "New Web
Service" import ignores that file, so you'd need to type in the build/start commands
and `EMBEDDING_PROVIDER=openai` by hand under Environment.

The other thing worth knowing: the free instance spins down after 15 minutes of no
traffic, so the first request after a quiet stretch takes 30-60 seconds. Everything
after that responds normally.

## Testing

```bash
pytest
```

17 tests, no network access or API key needed. Retrieval is tested against the real
sample document both on the dense side (a deterministic fake embedder) and the keyword
side (BM25 needs no embeddings at all, so it's tested against the real thing directly),
plus one test that goes through the actual fused hybrid retriever end to end. The
agent's logic (history handling, node skipping, retries, rate limiting, the actual
prompt content) is tested against a fake chat model that records what it was called
with — see `tests/fakes.py`. What this doesn't cover is whether a real OpenAI model's
answers are actually faithful to the retrieved context — that's a property of the model
and the prompt, not something a unit test can verify, which is what the real transcript
in `sample_io.md` is for.

## Trade-offs made and why

**FAISS instead of pgvector/Chroma.** One small document, one process, nothing writing
to the index concurrently — there's no reason to stand up a database for this. That
stops being true the moment there's more than one document set or more than one writer,
and pgvector would be the first thing I'd reach for then.

**Hybrid (BM25 + dense) retrieval, not dense alone.** A document this size and this
structured (short, clearly-sectioned policy text with specific numbers and names in it)
is exactly where pure dense retrieval can lose precision — an embedding model
represents "10% restocking fee" and "15% restocking fee" as very similar vectors, but
BM25 will happily tell them apart. Fusing the two via reciprocal rank fusion
(`EnsembleRetriever`, weighted 50/50 — see `scripts/vectorstore.py`) gets both exact-term
precision and paraphrase recall without picking one and losing the other. The 50/50
split is a reasonable starting point, not a tuned value — there's no eval set yet to tune
it against (see "What I'd add" below). BM25 itself is rebuilt from the source document
at every process start rather than persisted, since it has no model or training step to
amortize.

**Local embeddings, hosted generation.** They don't have to come from the same
provider. A small local model handles retrieval fine for a document this size, and
generation is where model quality actually matters, so that's where the API spend goes.
The trade-off is retrieval quality bounded by a more general-purpose embedding model —
worth re-benchmarking against `text-embedding-3-small` for a bigger or more
paraphrase-heavy corpus.

**A script for ingestion, not a `POST /ingest` endpoint.** The assignment left this
open. Ingestion is infrequent and can be slow, so it made more sense as a batch step
than something competing with chat traffic on the same process — and it keeps the API
surface at exactly the one endpoint the assignment asks for.

**No groundedness-check node.** Same reasoning as above — a second LLM call and a
branch to catch something the generation prompt (context-only framing, explicit
refusal instruction) already mostly handles for one small static document. Real
limitation: prompt compliance is a strong nudge, not a guarantee.

**In-memory session history, capped at 20 turns.** Good enough for the assignment's
scope. It doesn't survive a restart and doesn't work across multiple instances — both
are one-line swaps to Redis or Postgres, which is what I'd do first for a real
deployment. The rate limiter (above) makes the same trade-off for the same reason.

**No `create_react_agent`**, per the assignment. The graph is hand-built with
`StateGraph` instead — three nodes is little enough that any prebuilt scaffolding would
add more abstraction than it removes.

## What I'd add with more time

Not implemented, since the assignment isn't graded on feature completeness — but worth
naming honestly:

- A real eval set (question/expected-answer pairs) and something like an LLM-as-judge
  to catch regressions whenever the prompt or retrieval logic changes.
- Actual observability instead of JSON lines to stdout — OpenTelemetry or Langfuse,
  plus token/cost tracking per request.
- A reranking pass over the fused hybrid results before the top-k make it into the
  prompt (e.g. a cross-encoder), plus an eval set to actually tune the BM25/dense weight
  instead of leaving it at an untested 50/50. Fine as-is for one short document, worth
  revisiting the moment retrieval quality varies a lot by query.
- More than a single "ignore instructions in retrieved content" line for prompt
  injection and PII — scanning documents and model output before they reach a user.
- A shared session store (Redis/Postgres) so history survives a restart and works
  behind more than one instance.
