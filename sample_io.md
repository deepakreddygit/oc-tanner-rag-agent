# Sample input/output

This is a real transcript captured from a live run of this service (Python 3.10, OpenAI
`gpt-4o-mini`, local HuggingFace embeddings, FAISS), via the interactive docs at
`http://127.0.0.1:8000/docs`, all in one session (`session_id: "test"`) after running
`python scripts/ingest.py`. It covers the three required behaviors: a factual question
answered from the document, a summary request synthesized rather than quoted, and a
question the document can't answer.

The same three behaviors were re-verified against the public deployment at
https://oc-tanner-rag-agent.onrender.com/docs (OpenAI embeddings there instead of
HuggingFace -- see README "Deployment"), with matching results.

## 1. Factual question

```
POST /chat
{
  "session_id": "test",
  "message": "What is the refund window, and is there a restocking fee?"
}
```

```
{
  "session_id": "test",
  "response": "Customers can request a refund within 30 days of the delivery date. A 10% restocking fee applies to opened electronics priced over $500, but it does not apply to unopened electronics, items outside the Electronics category, or items that arrived defective or damaged in transit."
}
```

## 2. Summary request (synthesized, not quoted)

```
POST /chat
{
  "session_id": "test",
  "message": "Can you summarize the shipping policy?"
}
```

```
{
  "session_id": "test",
  "response": "Acme Corp ships to all 50 U.S. states and Canada. Standard shipping within the continental U.S. is free for orders over $50 and takes 3-5 business days; orders under $50 incur a $5.99 fee. Expedited shipping is available for an additional cost. Shipping to Alaska, Hawaii, and Canada may take 5-10 business days and may include extra carrier surcharges. Acme Corp does not ship outside the U.S. and Canada."
}
```

## 3. Insufficient information (explicit refusal, not speculation)

```
POST /chat
{
  "session_id": "test",
  "message": "Do you ship to Germany, and are there customs fees?"
}
```

```
{
  "session_id": "test",
  "response": "Based on the available documents, Acme Corp does not ship outside the United States and Canada, so I can't answer your question about shipping to Germany or customs fees."
}
```
