from fastapi.testclient import TestClient
from app.main import app
from forecasting.core import SeasonalNaive, metrics

def test_health():
    assert TestClient(app).get("/healthz").json()["status"]=="ok"

def test_baseline():
    assert SeasonalNaive(3).fit([1,2,3,4,5,6]).predict(4).tolist()==[4,5,6,4]

def test_metrics():
    assert metrics([1,2,3],[1,2,3])["mae"]==0
