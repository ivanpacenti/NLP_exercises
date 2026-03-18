Text-to-query (Graph-RAG style)
===============================

Purpose
-------
Build a **text-to-SPARQL system** over a small knowledge graph using a ReAct-like approach.

Students will:
- convert natural language → SPARQL
- query a local QLever instance
- expose functionality via a FastAPI web service


Background
----------
You are given:

- A Wikibase: *Keyboards Wikibase*  
- Exported RDF dataset: `keyboards.nt`

SPARQL prefixes:

```sparql
PREFIX kb: <https://keyboards.wikibase.cloud/entity/>
PREFIX kbt: <https://keyboards.wikibase.cloud/prop/direct/>
```

Task
----
Implement a FastAPI web service that:
- Set up a QLever graph database with 'keyboards' data
- Takes a natural language query (English or Danish)
- Converts it into SPARQL
- Executes it against QLever
- Returns the result as JSON

QLever
------
Installation
`pip install qlever`

Set up a Qleverfile, e.g.,
```
[data]
NAME = keyboards
DESCRIPTION = Keyboards Wikibase snapshot 

[index]
INPUT_FILES = keyboards.nt
CAT_INPUT_FILES = cat keyboards.nt

[server]
PORT = 7070
TIMEOUT = 30s

[runtime]
SYSTEM = docker
IMAGE = docker.io/adfreiburg/qlever:latest

[ui]
UI_CONFIG = default
UI_PORT = 8176
UI_SYSTEM = docker
UI_IMAGE = docker.io/adfreiburg/qlever-ui
```
Index and start the QLever server
```
qlever index
qlever start
``` 
and check that is running on port 7070.


API Specification
-----------------
Endpoint

`POST /v1/query`

Input
```
{
  "text": "What is the width of a Yamaha P-150?"
}
```
Output (simple version)
```
{
  "sparql": "...",
  "results": [{"width": 1385"}]
}
```
Output (recommended)
```
{
  "query": "What is the width of a Yamaha P-150?",
  "items_as_strings": ["Yamaha P-150"],
  "properties_as_strings": ["width"],
  "sparql": "...",
  "results": [
    {"width": 1385}
  ]
}
```

Possible Components (Tools)
---------------------------

Implement functions usable in a ReAct loop.

### Entity extraction (LLM-based)
```
def extract_entities(text: str) -> list[str]:
    ...
```

Example:
```
extract_entities("What is the width of a Yamaha P-150?")
# → {'items': ["Yamaha P-150"], 'properties': ["width"]}
```

### Item lookup (SPARQL)
```
def lookup_item(label: str) -> str:
    """Return QID (kb:Q...)"""
``` 
Query template:
```
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?item {
  ?item rdfs:label | skos:altLabel "Yamaha P-150"@en
}
```

### Property lookup
```
def lookup_property(label: str) -> str:
    """Return property PID (kbt:P...)"""
```

Query template:
```
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?property {
  ?property_item rdfs:label | skos:altLabel "width"@en ;
                 <http://wikiba.se/ontology#directClaim> ?property .
}
```

### SPARQL execution
```
def run_sparql(query: str) -> dict:
    ...
```

Endpoint:
`http://localhost:7070`


### Query construction (baseline)

Example:

Input:

```
"What is the width of a Yamaha P-150?"
```

Construct:
```
SELECT ?value WHERE {
  kb:Q1 kbt:P2 ?value .
}
```

Minimal Baseline Implementation
-------------------------------

``` 
def text_to_query(text):
    entities = extract_entities(text)
    item = lookup_item(entities[0])
    prop = lookup_property("width")

    sparql = f"""
    SELECT ?value WHERE {{
      {item} {prop} ?value .
    }}
    """

    return sparql
```

ReAct "Loop" Example
--------------------
```
Thought: Need entity
Action: extract_entities
Observation: ["Yamaha P-150"]

Thought: Resolve entity
Action: lookup_item
Observation: kb:Q1

Thought: Resolve property
Action: lookup_property
Observation: kbt:P2

Thought: Build query
Action: run_sparql
```

FastAPI Skeleton
----------------
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryInput(BaseModel):
    text: str

@app.post("/v1/query")
def query_endpoint(q: QueryInput):
    entities = extract_entities(q.text)

    item = lookup_item(entities[0])
    prop = lookup_property("width")

    sparql = f"""
    SELECT ?width WHERE {{
      {item} {prop} ?width .
    }}
    """

    results = run_sparql(sparql)

    return {
        "query": q.text,
        "entities": entities,
        "sparql": sparql,
        "results": results
    }
```

Tests
-----
```
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

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
```

Example Queries
---------------
- "What is the width of a Yamaha P-150?"
- "What is the height of a Yamaha P-150?"
- "Hvad er bredden på Yamaha P-150?"

Advanced:
- "What is the smallest Swedish keyboard?"

Optional (advanced)
-------------------
- Use DSPy.ReAct
- Add multilingual prompting
- Use embeddings for retrieval
- Use query examples stored in Wikibase

Discussion Questions
--------------------
- What are advantages of: rule-based vs LLM-based parsing?
- How reliable is entity linking?
- What happens when multiple matches exist?
- How to handle Danish vs English?
- How would you scale this to full Wikidata?

Requirements & Resources
------------------------
- Python
- FastAPI
- Requests
- QLever endpoint
- CampusAI LLM API
- Text-to-query
  - Knowledge graphs - Chapter "Text-to-query"
- Entity linking
  - Natural language processing - Chapter "Entity linking"
- Prompt engineering
  - Natural language processing - Chapter "Prompt engineering".
- ReAct (for optional advanced)
  - Natural language processing - Section 8.5 ReAct prompting.
  - Scientific literature 
    - SPINACH: SPARQL-Based Information Navigation for Challenging Real-World Questions
    - GRASP: Generic Reasoning And SPARQL Generation Across Knowledge Graphs


Deliverables
------------
A zipped repository in root (git archive -o latest.zip HEAD)