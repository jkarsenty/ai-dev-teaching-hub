import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app



def test_healthcheck():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_route_ok():
    client = app.test_client()
    payload = {"text": "je suis très content"}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.get_json()

    assert "label" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_predict_missing_text_field():
    client = app.test_client()
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
