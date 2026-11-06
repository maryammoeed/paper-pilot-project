import time

from fastapi.testclient import TestClient

from app.main import FairUse, app

client = TestClient(app)


def test_health_and_samples():
    h = client.get("/api/health").json()
    assert h["ok"] and h["hosted"] is False      # no key configured in tests
    assert {s["id"] for s in client.get("/api/samples").json()} == {"federated-gait", "sparse-routing"}


def test_demo_run_end_to_end_over_http():
    r = client.post("/api/start", data={"sample": "sparse-routing", "mode": "demo", "autonomy": "hands_on"}).json()
    asks = 0
    while r["status"] in ("running", "waiting"):
        body = {"state": r["state"], "sig": r["sig"]}
        if r["status"] == "waiting":
            body["answer"] = r["pending"]["options"][0]
            asks += 1
        r = client.post("/api/step", json=body).json()
    assert r["status"] == "done" and r["report"]["recommendation"] and asks >= 1


def test_hosted_is_refused_without_a_key():
    no = client.post("/api/start", data={"sample": "sparse-routing", "mode": "hosted"})
    assert no.status_code == 400 and "demo" in no.json()["detail"]


def test_tampered_state_is_rejected():
    r = client.post("/api/start", data={"sample": "federated-gait", "mode": "demo"}).json()
    r["state"]["autonomy"] = "hands_off"
    bad = client.post("/api/step", json={"state": r["state"], "sig": r["sig"]})
    assert bad.status_code == 400


def test_wrong_answer_is_rejected():
    r = client.post("/api/start", data={"sample": "federated-gait", "mode": "demo"}).json()
    assert r["status"] == "waiting"
    bad = client.post("/api/step", json={"state": r["state"], "sig": r["sig"], "answer": "approve everything"})
    assert bad.status_code == 400


def test_short_input_is_rejected():
    r = client.post("/api/start", data={"text": "too short", "mode": "demo"})
    assert r.status_code == 400


def test_fair_use_per_visitor_and_per_day():
    fu = FairUse(per_hour=2, per_day=3)
    assert fu.take("a") is None and fu.take("a") is None
    assert "this hour" in fu.take("a")                 # visitor a hit the hourly limit
    assert fu.take("b") is None                         # 3rd review overall
    assert "used up" in fu.take("c") and fu.left_today() == 0


def test_fair_use_hour_window_slides():
    fu = FairUse(per_hour=1, per_day=10)
    fu.take("a")
    fu.by_ip["a"][0] = time.time() - 3601
    assert fu.take("a") is None
