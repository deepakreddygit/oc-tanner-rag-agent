from fastapi.testclient import TestClient

from app.agent import Agent
from app.main import app, get_agent
from tests.fakes import RecordingFakeChatModel


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "vectorstore_ready" in resp.json()


def test_chat_endpoint_returns_grounded_answer(policy_vectorstore):
    fake_llm = RecordingFakeChatModel(
        responses=["A 10% restocking fee applies to opened electronics over $500."]
    )
    app.dependency_overrides[get_agent] = lambda: Agent(llm=fake_llm, vectorstore=policy_vectorstore, k=4)
    client = TestClient(app)
    try:
        resp = client.post(
            "/chat", json={"session_id": "api-test-1", "message": "Is there a restocking fee?"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "api-test-1"
        assert "restocking fee" in body["response"].lower()
    finally:
        app.dependency_overrides.clear()


def test_chat_endpoint_uses_session_history_on_second_turn(policy_vectorstore):
    fake_llm = RecordingFakeChatModel(
        responses=[
            "You have 30 days from delivery to return an item.",
            "Is there a restocking fee on that return?",
            "A 10% restocking fee applies to opened electronics over $500.",
        ]
    )
    app.dependency_overrides[get_agent] = lambda: Agent(llm=fake_llm, vectorstore=policy_vectorstore, k=4)
    client = TestClient(app)
    try:
        r1 = client.post(
            "/chat", json={"session_id": "api-test-2", "message": "What is the refund window?"}
        )
        assert r1.json()["response"] == "You have 30 days from delivery to return an item."

        r2 = client.post(
            "/chat", json={"session_id": "api-test-2", "message": "Is there a restocking fee?"}
        )
        assert r2.status_code == 200
        assert r2.json()["response"] == "A 10% restocking fee applies to opened electronics over $500."
        # turn 1: 1 call (generate only). turn 2: 2 calls (contextualize + generate).
        assert len(fake_llm.calls) == 3
    finally:
        app.dependency_overrides.clear()


def test_chat_rejects_empty_message(policy_vectorstore):
    # Overridden so this test isolates request validation from agent availability --
    # without a real vector store (or an override) the agent dependency itself would
    # 503 before validation ever runs, which is not what this test is checking.
    fake_llm = RecordingFakeChatModel(responses=["unused"])
    app.dependency_overrides[get_agent] = lambda: Agent(llm=fake_llm, vectorstore=policy_vectorstore, k=4)
    client = TestClient(app)
    try:
        resp = client.post("/chat", json={"session_id": "api-test-3", "message": ""})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_chat_returns_503_when_vectorstore_missing(monkeypatch):
    # Forces the "ingestion hasn't run yet" branch regardless of whether this machine
    # actually has a vector store on disk, so the test is deterministic in CI and on a
    # freshly cloned checkout alike.
    import app.main as main_module

    monkeypatch.setattr(main_module, "vectorstore_exists", lambda: False)
    monkeypatch.setattr(main_module, "_agent", None)
    client = TestClient(app)
    resp = client.post("/chat", json={"session_id": "api-test-4", "message": "hello"})
    assert resp.status_code == 503
    assert "ingest.py" in resp.json()["detail"]
