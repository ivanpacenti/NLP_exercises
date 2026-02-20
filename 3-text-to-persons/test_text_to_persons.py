import json
import app

def test_examples_from_prompt(monkeypatch):
    def fake_send_message(prompt):
        return {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "persons": ["Mette Frederiksen"]
                    })
                }
            }]
        }

    monkeypatch.setattr(app, "send_message", fake_send_message)

    result = app.extract_entities("Ms Mette Frederiksen is in New York today.")
    assert result == {"persons": ["Mette Frederiksen"]}