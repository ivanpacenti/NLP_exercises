import json
import re

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from campus_ai_api import send_message

SPARQL_ENDPOINT = "http://localhost:7070/sparql"
app = FastAPI()

# Prompt used to extract item names and properties from a natural language question.
PROMPT_TEMPLATE = (
    "For a text-to-SPARQL system over a keyboard knowledge graph, extract the item names and property names from the question. "
    "Return ONLY valid JSON with the schema: {{\"items\":[...], \"properties\":[...]}}. "
    "Do not add code fences or extra text. "
    "Put the most likely match first and include short useful variants after it when relevant. "
    "Use English property names when possible, even if the question is in Danish. "
    "Example response: {{\"items\":[\"Yamaha P-150\"],\"properties\":[\"width\"]}}. "
    "If no item or property is found, return empty lists for that field.\n\n"
    "Text:\n"
    "{text}"
)

def _extract_content(response: dict) -> str:
    # CampusAI responses can have slightly different shapes.
    if isinstance(response, dict):
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"]
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        if "error" in response:
            raise ValueError(f"CampusAI error: {response['error']}")
    raise ValueError(f"Unexpected CampusAI response structure: {response}")

def _parse_entities(content: str) -> dict[str, list[str]]:
    # Parse the LLM output and keep only items and properties.
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*}", content, re.S)
        if not match:
            raise ValueError("No JSON object found in response")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Invalid entities field in response")
    cleaned = {"items": [], "properties": []}
    for key in ("items", "properties"):
        value = data.get(key, [])
        if isinstance(value, list):
            cleaned[key] = [str(v) for v in value if str(v).strip()]
    return cleaned

def extract_entities(text: str) -> dict[str, list[str]]:
    prompt = PROMPT_TEMPLATE.format(text=text)
    # Ask the LLM to extract candidate items and properties.
    response = send_message(prompt)

    content = _extract_content(response)
    try:
        parsed = _parse_entities(content)
    except Exception as exc:
        raise ValueError(f"Failed to parse model output: {content!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed entities is not a dict: {parsed!r}")
    return parsed


def _resolve_first_match(candidates: list[str], resolver) -> tuple[str, str]:
    for candidate in candidates:
        resolved = resolver(candidate)
        if resolved:
            return candidate, resolved
    return "", ""


def _build_simple_value_query(qid: str, pid: str, variable_name: str = "value") -> str:
    # Baseline SPARQL template for simple attribute questions.
    return f"""
PREFIX kb: <https://keyboards.wikibase.cloud/entity/>
PREFIX kbt: <https://keyboards.wikibase.cloud/prop/direct/>

SELECT ?{variable_name} WHERE {{
  kb:{qid} kbt:{pid} ?{variable_name} .
}}
"""


def _escape_sparql_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _simplify_bindings(bindings: list[dict]) -> list[dict[str, str]]:
    simplified = []
    for row in bindings:
        simplified_row = {}
        for key, value in row.items():
            if isinstance(value, dict) and "value" in value:
                simplified_row[key] = value["value"]
        simplified.append(simplified_row)
    return simplified

# Get only the entity/property id from the full URI
def _extract_id_from_uri(uri: str) -> str:
    if not uri:
        return ""
    return uri.rstrip("/").rsplit("/", 1)[-1]

# Lowercase the text and remove extra spaces
def _normalize_lookup_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _dedupe_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    # Keep only one candidate per entity/property id.
    deduped = []
    seen_ids = set()
    for candidate in candidates:
        candidate_id = candidate.get("id", "")
        if not candidate_id or candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        deduped.append(candidate)
    return deduped


def _candidate_score(candidate: dict[str, str], query_text: str) -> tuple[int, int, str]:
    # Prefer exact label matches, then prefer entities with more statements in the graph.
    normalized_query = _normalize_lookup_text(query_text)
    normalized_label = _normalize_lookup_text(candidate.get("label", ""))
    exact_label_match = int(normalized_label == normalized_query)
    statement_count = int(float(candidate.get("statement_count", "0") or 0))
    candidate_id = candidate.get("id", "")
    return exact_label_match, statement_count, candidate_id


def _pick_best_candidate(candidates: list[dict[str, str]], query_text: str) -> str:
    if not candidates:
        return ""
    ranked = sorted(
        _dedupe_candidates(candidates),
        key=lambda candidate: _candidate_score(candidate, query_text),
        reverse=True,
    )
    return ranked[0].get("id", "")


def run_sparql(query: str) -> list[dict[str, str]]:
    # Send the SPARQL query to the local QLever endpoint.
    with httpx.Client(timeout=20) as client:
        response = client.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
        )
        response.raise_for_status()
    payload = response.json()
    bindings = payload.get("results", {}).get("bindings", [])
    return _simplify_bindings(bindings)


def lookup_item_candidates(text: str, language: str = "en") -> list[dict[str, str]]:
    # Find possible keyboard items that match the extracted text.
    escaped_text = _escape_sparql_string(text)
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?item ?label (COUNT(?p) AS ?statement_count) WHERE {{
  ?item rdfs:label | skos:altLabel "{escaped_text}"@{language} .
  FILTER(STRSTARTS(STR(?item), "https://keyboards.wikibase.cloud/entity/Q"))
  OPTIONAL {{
    ?item rdfs:label ?label .
    FILTER(LANG(?label) = "{language}")
  }}
  OPTIONAL {{
    ?item ?p ?o .
  }}
}}
GROUP BY ?item ?label
"""
    results = run_sparql(sparql)
    return [
        {
            "id": _extract_id_from_uri(result.get("item", "")),
            "label": result.get("label", ""),
            "statement_count": result.get("statement_count", "0"),
        }
        for result in results
    ]


def lookup_item(text: str, language: str = "en") -> str:
    return _pick_best_candidate(lookup_item_candidates(text, language), text)


def lookup_property_candidates(text: str, language: str = "en") -> list[dict[str, str]]:
    # Find possible properties that match the extracted text.
    escaped_text = _escape_sparql_string(text)
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT ?property ?label (COUNT(?statement_property) AS ?statement_count) WHERE {{
  ?property_item rdfs:label | skos:altLabel "{escaped_text}"@{language} ;
                 wikibase:directClaim ?property .
  OPTIONAL {{
    ?property_item rdfs:label ?label .
    FILTER(LANG(?label) = "{language}")
  }}
  OPTIONAL {{
    ?property_item ?statement_property ?statement_value .
  }}
}}
GROUP BY ?property ?label
"""
    results = run_sparql(sparql)
    return [
        {
            "id": _extract_id_from_uri(result.get("property", "")),
            "label": result.get("label", ""),
            "statement_count": result.get("statement_count", "0"),
        }
        for result in results
    ]


def lookup_property(text: str, language: str = "en") -> str:
    return _pick_best_candidate(lookup_property_candidates(text, language), text)


def text_to_query(text: str) -> dict[str, object]:
    # Full baseline pipeline: extract entities, resolve them, build SPARQL, run it.
    entities = extract_entities(text)
    item_label, qid = _resolve_first_match(entities.get("items", []), lookup_item)
    property_label, pid = _resolve_first_match(entities.get("properties", []), lookup_property)

    if not item_label:
        raise ValueError("Could not resolve any item from the extracted entities")
    if not property_label:
        raise ValueError("Could not resolve any property from the extracted entities")

    sparql = _build_simple_value_query(qid, pid)
    results = run_sparql(sparql)
    flat_entities = entities.get("items", []) + entities.get("properties", [])

    return {
        "query": text,
        "entities": flat_entities,
        "items_as_strings": entities.get("items", []),
        "properties_as_strings": entities.get("properties", []),
        "resolved_item": {"label": item_label, "qid": qid},
        "resolved_property": {"label": property_label, "pid": pid},
        "sparql": sparql,
        "results": results,
    }


class QueryInput(BaseModel):
    text: str


@app.post("/v1/query")
def query_endpoint(payload: QueryInput) -> dict[str, object]:
    try:
        return text_to_query(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"SPARQL endpoint error: {exc}") from exc
