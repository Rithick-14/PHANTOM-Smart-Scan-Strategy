"""
Unit tests for FastAPI REST API Microservice endpoints.
"""

from fastapi.testclient import TestClient
from smart_ew.api.app import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_api_reset_and_step():
    res_reset = client.post("/simulation/reset?num_bands=5&total_steps=50")
    assert res_reset.status_code == 200

    res_step = client.post("/simulation/step")
    assert res_step.status_code == 200
    data = res_step.json()
    assert data["step"] == 0
    assert 0 <= data["scanned_band"] < 5
