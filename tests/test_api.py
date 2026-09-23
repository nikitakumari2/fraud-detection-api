"""Run with: pytest  (after python train.py has created model/)"""

from fastapi.testclient import TestClient

from app.main import DEMO, RAW_COLS, app

client = TestClient(app)
ROW = DEMO.iloc[0][RAW_COLS].to_dict()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_probability():
    r = client.post("/predict", json=ROW)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_predict_explain_returns_top_features():
    r = client.post("/predict?explain=true", json=ROW)
    assert len(r.json()["top_features"]) == 5


def test_missing_field_is_rejected():
    bad = {k: v for k, v in ROW.items() if k != "V14"}
    assert client.post("/predict", json=bad).status_code == 422


def test_demo_and_monitoring():
    client.post("/reset")
    r = client.get("/demo/next?n=60")
    assert len(r.json()["transactions"]) == 60
    m = client.get("/monitoring").json()
    assert m["scored"] == 60 and m["drift"]["window"] == 60
