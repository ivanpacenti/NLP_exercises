import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from wikidata_models import (
    AllResponse,
    BirthdayResponse,
    PersonRequest,
    PoliticalPartyResponse,
    StudentsResponse,
    SupervisorResponse,
)

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "person-to-wikidata/1.0 (student project)"
TIMEOUT_SECONDS = float(10)


app = FastAPI()


async def _search_qid(name: str) -> str:
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json",
        "type": "item",
        "limit": 1,
    }
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=headers) as client:
        resp = await client.get(WIKIDATA_SEARCH_URL, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Wikidata search failed: {resp.status_code}")
    data = resp.json()
    results = data.get("search") or []
    if not results:
        raise HTTPException(status_code=404, detail="No matching Wikidata entity found")
    return results[0]["id"]


async def _sparql_select(query: str) -> List[Dict[str, Any]]:
    headers = {"Accept": "application/sparql+json", "User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=headers) as client:
        resp = await client.get(WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Wikidata SPARQL failed: {resp.status_code}")
    data = resp.json()
    return data.get("results", {}).get("bindings", [])


async def get_birthday(qid: str) -> Optional[str]:
    query = f"""
    SELECT ?dob WHERE {{
      wd:{qid} wdt:P569 ?dob .
    }} LIMIT 1
    """
    rows = await _sparql_select(query)
    if not rows:
        return None
    value = rows[0]["dob"]["value"]
    return value.split("T")[0]


async def get_students(qid: str) -> List[Dict[str, str]]:
    query = f"""
    SELECT ?student ?studentLabel WHERE {{
      ?student wdt:P184 wd:{qid} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    rows = await _sparql_select(query)
    students = []
    for row in rows:
        uri = row["student"]["value"]
        qid_val = uri.rsplit("/", 1)[-1]
        label = row["studentLabel"]["value"]
        students.append({"label": label, "qid": qid_val})
    return students


async def get_political_party(qid: str) -> List[Dict[str, str]]:
    query = f"""
    SELECT ?party ?partyLabel WHERE {{
      wd:{qid} wdt:P102 ?party .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    rows = await _sparql_select(query)
    parties = []
    for row in rows:
        uri = row["party"]["value"]
        qid_val = uri.rsplit("/", 1)[-1]
        label = row["partyLabel"]["value"]
        parties.append({"label": label, "qid": qid_val})
    return parties


async def get_supervisors(qid: str) -> List[Dict[str, str]]:
    query = f"""
    SELECT ?supervisor ?supervisorLabel WHERE {{
      wd:{qid} wdt:P184 ?supervisor .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    rows = await _sparql_select(query)
    supervisors = []
    for row in rows:
        uri = row["supervisor"]["value"]
        qid_val = uri.rsplit("/", 1)[-1]
        label = row["supervisorLabel"]["value"]
        supervisors.append({"label": label, "qid": qid_val})
    return supervisors


@app.post("/v1/birthday", response_model=BirthdayResponse)
async def birthday(req: PersonRequest):
    qid = await _search_qid(req.person)
    dob = await get_birthday(qid)
    return {"person": req.person, "qid": qid, "birthday": dob}


@app.post("/v1/students", response_model=StudentsResponse)
async def students(req: PersonRequest):
    qid = await _search_qid(req.person)
    students_list = await get_students(qid)
    return {"person": req.person, "qid": qid, "students": students_list}


@app.post("/v1/all", response_model=AllResponse)
async def all_info(req: PersonRequest):
    qid = await _search_qid(req.person)
    dob = await get_birthday(qid)
    students_list = await get_students(qid)
    return {"person": req.person, "qid": qid, "birthday": dob, "students": students_list}


@app.post("/v1/political-party", response_model=PoliticalPartyResponse)
async def political_party(req: PersonRequest):
    qid = await _search_qid(req.person)
    parties = await get_political_party(qid)
    return {"person": req.person, "qid": qid, "political_party": parties}


@app.post("/v1/supervisor", response_model=SupervisorResponse)
async def supervisor(req: PersonRequest):
    qid = await _search_qid(req.person)
    supervisors = await get_supervisors(qid)
    return {"person": req.person, "qid": qid, "supervisors": supervisors}
