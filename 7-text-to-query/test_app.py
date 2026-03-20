from fastapi.testclient import TestClient

import app as text_to_query_app


client = TestClient(text_to_query_app.app)


def test_query_endpoint_returns_sparql_and_results(monkeypatch):
    monkeypatch.setattr(
        text_to_query_app,
        "extract_entities",
        lambda text: {"items": ["Yamaha P-150"], "properties": ["width"]},
    )
    monkeypatch.setattr(text_to_query_app, "lookup_item", lambda text, language="en": "Q1")
    monkeypatch.setattr(text_to_query_app, "lookup_property", lambda text, language="en": "P2")
    monkeypatch.setattr(text_to_query_app, "run_sparql", lambda query: [{"value": "1385"}])

    response = client.post(
        "/v1/query",
        json={"text": "What is the width of a Yamaha P-150?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is the width of a Yamaha P-150?"
    assert data["items_as_strings"] == ["Yamaha P-150"]
    assert data["properties_as_strings"] == ["width"]
    assert data["resolved_item"]["qid"] == "Q1"
    assert data["resolved_property"]["pid"] == "P2"
    assert "SELECT ?value WHERE" in data["sparql"]
    assert data["results"] == [{"value": "1385"}]


def test_query_endpoint_returns_400_when_entity_resolution_fails(monkeypatch):
    monkeypatch.setattr(
        text_to_query_app,
        "extract_entities",
        lambda text: {"items": [], "properties": ["width"]},
    )

    response = client.post(
        "/v1/query",
        json={"text": "What is the width of an unknown keyboard?"},
    )

    assert response.status_code == 400
    assert "Could not resolve any item" in response.json()["detail"]



def test_width_query():
    response = client.post("/v1/query", json={
        "text": "What is the width of a Yamaha P-150?"
    })

    assert response.status_code == 200
    data = response.json()

    assert "Yamaha P-150" in data["entities"]
    assert "sparql" in data
    assert "results" in data

if __name__ == "__main__":
    test_width_query()
    print("All tests passed!")