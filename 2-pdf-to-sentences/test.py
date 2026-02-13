

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_extract_sentences():
    # Adjust the path to your PDF file
    pdf_path = "2303.15133.pdf"

    with open(pdf_path, "rb") as f:
        files = {"pdf_file": ("2303.15133.pdf", f, "application/pdf")}
        response = client.post("/v1/extract-sentences", files=files)

    # --- basic checks
    assert response.status_code == 200, response.text
    data = response.json()
    assert "sentences" in data
    assert isinstance(data["sentences"], list)

    sentence = "How language should best be handled is not clear."
    assert sentence in data["sentences"]