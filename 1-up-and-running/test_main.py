from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_positive_sentiment():
    response = client.post("/v1/sentiment", json={"text": "Det var en god lærer."})
    assert response.status_code == 200
    assert response.json() == {"score": 3}

def test_negative_sentiment():
    response = client.post("/v1/sentiment", json={"text": "It was a bad course"})
    assert response.status_code == 200
    assert response.json() == {"score": -3}


if __name__ == "__main__":
    test_positive_sentiment()
    test_negative_sentiment()
    print("All tests passed!")