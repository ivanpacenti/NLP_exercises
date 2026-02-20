import json
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from campus_ai_api import send_message


app = FastAPI()


class ExtractRequest(BaseModel):
    text: str


ResponseModel = dict[str, list[str]]

PROMPT_TEMPLATE = (
    "Extract all named entities from the text and group them by category. "
    "Return ONLY valid JSON with the schema: {{\"persons\":[...], \"gpe\":[...], ...}}. "
    "Do not add code fences or extra text. "
    "Example response: {{\"persons\":[\"Mario Rossi\"],\"gpe\":[\"Roma\"]}}. "
    "Use standard NER category names (e.g., persons, org, gpe, loc, date, time, money, "
    "percent, product, event, work_of_art, law, language, norp, fac, quantity, ordinal, cardinal). "
    "If a category is empty, omit it. If no entities are found, return {{}}.\n\n"
    "Text:\n"
    "{text}"
)


def _parse_entities(content: str) -> dict[str, list[str]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*}", content, re.S)
        if not match:
            raise ValueError("No JSON object found in response")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Invalid entities field in response")
    cleaned = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, list):
            cleaned[key] = [str(v) for v in value]
    return cleaned

#parses the campusai json response, which is OpanAi compatible
def _extract_content(response: dict) -> str:
    if isinstance(response, dict):
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"]
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        if "error" in response:
            raise ValueError(f"CampusAI error: {response['error']}")
    raise ValueError(f"Unexpected CampusAI response structure: {response}")


def extract_entities(text: str) -> dict[str, list[str]]:
    prompt = PROMPT_TEMPLATE.format(text=text)
    #api call
    response = send_message(prompt)

    content = _extract_content(response)
    try:
        parsed = _parse_entities(content)
    except Exception as exc:
        raise ValueError(f"Failed to parse model output: {content!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed entities is not a dict: {parsed!r}")
    return parsed


@app.post("/v1/extract-persons", response_model=ResponseModel)
def extract_persons_endpoint(req: ExtractRequest):
    try:
        entities = extract_entities(req.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return entities
