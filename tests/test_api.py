from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_home_route():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_route():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["pipeline_loaded"] is True
    assert data["number_of_raw_features"] > 0


def test_features_route():
    response = client.get("/features")

    assert response.status_code == 200

    data = response.json()

    assert "required_raw_features" in data
    assert len(data["required_raw_features"]) > 0


def test_predict_missing_features_fails():
    response = client.post(
        "/predict",
        json={
            "features": {
                "X0": "a"
            }
        }
    )

    assert response.status_code == 400
    assert "missing_features" in response.json()["detail"]