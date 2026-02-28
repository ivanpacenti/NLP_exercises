"""
DTU Entity Linking Frontend Web App
====================================

This frontend Web application is used to test an independently built
entity linking microservice.

The microservice exposes:

- POST /v1/birthday
- POST /v1/students
- POST /v1/all

This frontend:

- Submits the "input" part of a gold dataset (20 persons)
- Receives the Web service response
- Compares it against the expected output
- Computes automatic scoring
- Displays operational metrics
- Handles timeouts pedagogically
- Supports asynchronous concurrent requests

The interface is intentionally minimal:
- FastAPI
- Vanilla JavaScript
- Inline CSS
- No external frontend libraries

Design goals:
-------------
- Minimal dependencies
- Pedagogically readable
- Clear scoring logic
- Operational transparency

Requirements:
-------------
pip install fastapi uvicorn httpx

Run:
----
uvicorn person-to-wikidata-ui:app --reload --port 8001

"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import asyncio
import json
import random
import time

app = FastAPI()

SERVICE_BASE_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 6.0


# ---------------------------------------------------------------------
# GOLD DATASET (insert full 20 persons here)
# ---------------------------------------------------------------------

DATASET = [
  {
    "input": {
      "person": "Mette Frederiksen",
      "context": "When was Mette Frederiksen born?"
    },
    "output_birthday": {
      "person": "Mette Frederiksen",
      "qid": "Q5015",
      "birthday": "1977-11-19"
    },
    "output_students": {
      "person": "Mette Frederiksen",
      "qid": "Q5015",
      "students": []
    },
    "output_all": {
      "person": "Mette Frederiksen",
      "qid": "Q5015",
      "birthday": "1977-11-19",
      "students": []
    }
  },
  {
    "input": {
      "person": "Niels Bohr",
      "context": "Niels Bohr married Margrethe Nørlund."
    },
    "output_birthday": {
      "person": "Niels Bohr",
      "qid": "Q7085",
      "birthday": "1885-10-07"
    },
    "output_students": {
      "person": "Niels Bohr",
      "qid": "Q7085",
      "students": [
        {
          "label": "Aage Bohr",
          "qid": "Q103854"
        },
        {
          "label": "Oskar Klein",
          "qid": "Q251524"
        },
        {
          "label": "Lev Landau",
          "qid": "Q133267"
        },
        {
          "label": "John Archibald Wheeler",
          "qid": "Q202631"
        },
        {
          "label": "Hans Kramers",
          "qid": "Q451225"
        },
        {
          "label": "Arnold Sommerfeld",
          "qid": "Q77078"
        },
        {
          "label": "Robert Bruce Lindsay",
          "qid": "Q7342453"
        },
        {
          "label": "Svein Rosseland",
          "qid": "Q102415020"
        }
      ]
    },
    "output_all": {
      "person": "Niels Bohr",
      "qid": "Q7085",
      "birthday": "1885-10-07",
      "students": [
        {
          "label": "Aage Bohr",
          "qid": "Q103854"
        },
        {
          "label": "Oskar Klein",
          "qid": "Q251524"
        },
        {
          "label": "Lev Landau",
          "qid": "Q133267"
        },
        {
          "label": "John Archibald Wheeler",
          "qid": "Q202631"
        },
        {
          "label": "Hans Kramers",
          "qid": "Q451225"
        },
        {
          "label": "Arnold Sommerfeld",
          "qid": "Q77078"
        },
        {
          "label": "Robert Bruce Lindsay",
          "qid": "Q7342453"
        },
        {
          "label": "Svein Rosseland",
          "qid": "Q102415020"
        }
      ]
    }
  },
  {
    "input": {
      "person": "H. C. Andersen",
      "context": "In which year was H. C. Andersen born?"
    },
    "output_birthday": {
      "person": "H. C. Andersen",
      "qid": "Q5673",
      "birthday": "1805-04-02"
    },
    "output_students": {
      "person": "H. C. Andersen",
      "qid": "Q5673",
      "students": []
    },
    "output_all": {
      "person": "H. C. Andersen",
      "qid": "Q5673",
      "birthday": "1805-04-02",
      "students": []
    }
  },
  {
    "input": {
      "person": "Margrethe II of Denmark",
      "context": "Margrethe II of Denmark became queen in 1972."
    },
    "output_birthday": {
      "person": "Margrethe II of Denmark",
      "qid": "Q102139",
      "birthday": "1940-04-16"
    },
    "output_students": {
      "person": "Margrethe II of Denmark",
      "qid": "Q102139",
      "students": []
    },
    "output_all": {
      "person": "Margrethe II of Denmark",
      "qid": "Q102139",
      "birthday": "1940-04-16",
      "students": []
    }
  },
  {
    "input": {
      "person": "Kierkegaard",
      "context": "Kierkegaard influenced existentialism."
    },
    "output_birthday": {
      "person": "Søren Kierkegaard",
      "qid": "Q6512",
      "birthday": "1813-05-05"
    },
    "output_students": {
      "person": "Søren Kierkegaard",
      "qid": "Q6512",
      "students": []
    },
    "output_all": {
      "person": "Søren Kierkegaard",
      "qid": "Q6512",
      "birthday": "1813-05-05",
      "students": []
    }
  },
  {
    "input": {
      "person": "Tycho Brahe",
      "context": "Tycho Brahe made astronomical observations."
    },
    "output_birthday": {
      "person": "Tycho Brahe",
      "qid": "Q36620",
      "birthday": "1546-12-24"
    },
    "output_students": {
      "person": "Tycho Brahe",
      "qid": "Q36620",
      "students": [
        {
          "label": "David Gans",
          "qid": "Q1174512"
        },
        {
          "label": "Adriaan Metius",
          "qid": "Q367638"
        },
        {
          "label": "Ambrosius Rhode",
          "qid": "Q459735"
        },
        {
          "label": "Paul Wittich",
          "qid": "Q88089"
        },
        {
          "label": "Peter Crüger",
          "qid": "Q73013"
        },
        {
          "label": "Simon Marius",
          "qid": "Q76684"
        },
        {
          "label": "Johannes Kepler",
          "qid": "Q8963"
        }
      ]
    },
    "output_all": {
      "person": "Tycho Brahe",
      "qid": "Q36620",
      "birthday": "1546-12-24",
      "students": [
        {
          "label": "David Gans",
          "qid": "Q1174512"
        },
        {
          "label": "Adriaan Metius",
          "qid": "Q367638"
        },
        {
          "label": "Ambrosius Rhode",
          "qid": "Q459735"
        },
        {
          "label": "Paul Wittich",
          "qid": "Q88089"
        },
        {
          "label": "Peter Crüger",
          "qid": "Q73013"
        },
        {
          "label": "Simon Marius",
          "qid": "Q76684"
        },
        {
          "label": "Johannes Kepler",
          "qid": "Q8963"
        }
      ]
    }
  },
  {
    "input": {
      "person": "Aage Bohr",
      "context": "Aage Bohr received the Nobel Prize."
    },
    "output_birthday": {
      "person": "Aage Bohr",
      "qid": "Q103854",
      "birthday": "1922-06-19"
    },
    "output_students": {
      "person": "Aage Bohr",
      "qid": "Q103854",
      "students": []
    },
    "output_all": {
      "person": "Aage Bohr",
      "qid": "Q103854",
      "birthday": "1922-06-19",
      "students": []
    }
  },
  {
    "input": {
      "person": "Lars Løkke Rasmussen",
      "context": "When was Lars Løkke Rasmussen born?"
    },
    "output_birthday": {
      "person": "Lars Løkke Rasmussen",
      "qid": "Q182397",
      "birthday": "1964-05-15"
    },
    "output_students": {
      "person": "Lars Løkke Rasmussen",
      "qid": "Q182397",
      "students": []
    },
    "output_all": {
      "person": "Lars Løkke Rasmussen",
      "qid": "Q182397",
      "birthday": "1964-05-15",
      "students": []
    }
  },
  {
    "input": {
      "person": "Rasmussen",
      "context": "Rasmussen was Secretary General of NATO."
    },
    "output_birthday": {
      "person": "Anders Fogh Rasmussen",
      "qid": "Q46052",
      "birthday": "1953-01-26"
    },
    "output_students": {
      "person": "Anders Fogh Rasmussen",
      "qid": "Q46052",
      "students": []
    },
    "output_all": {
      "person": "Anders Fogh Rasmussen",
      "qid": "Q46052",
      "birthday": "1953-01-26",
      "students": []
    }
  },
  {
    "input": {
      "person": "Viggo Mortensen",
      "context": "Viggo Mortensen starred in The Lord of the Rings."
    },
    "output_birthday": {
      "person": "Viggo Mortensen",
      "qid": "Q171363",
      "birthday": "1958-10-20"
    },
    "output_students": {
      "person": "Viggo Mortensen",
      "qid": "Q171363",
      "students": []
    },
    "output_all": {
      "person": "Viggo Mortensen",
      "qid": "Q171363",
      "birthday": "1958-10-20",
      "students": []
    }
  },
  {
    "input": {
      "person": "Mads Mikkelsen",
      "context": "Mads Mikkelsen played Hannibal Lecter."
    },
    "output_birthday": {
      "person": "Mads Mikkelsen",
      "qid": "Q294647",
      "birthday": "1965-11-22"
    },
    "output_students": {
      "person": "Mads Mikkelsen",
      "qid": "Q294647",
      "students": []
    },
    "output_all": {
      "person": "Mads Mikkelsen",
      "qid": "Q294647",
      "birthday": "1965-11-22",
      "students": []
    }
  },
  {
    "input": {
      "person": "Wozniacki",
      "context": "Wozniacki won the Australian Open."
    },
    "output_birthday": {
      "person": "Caroline Wozniacki",
      "qid": "Q30767",
      "birthday": "1990-07-11"
    },
    "output_students": {
      "person": "Caroline Wozniacki",
      "qid": "Q30767",
      "students": []
    },
    "output_all": {
      "person": "Caroline Wozniacki",
      "qid": "Q30767",
      "birthday": "1990-07-11",
      "students": []
    }
  },
  {
    "input": {
      "person": "Peter Høeg",
      "context": "Peter Høeg wrote Smilla's Sense of Snow."
    },
    "output_birthday": {
      "person": "Peter Høeg",
      "qid": "Q337384",
      "birthday": "1957-05-17"
    },
    "output_students": {
      "person": "Peter Høeg",
      "qid": "Q337384",
      "students": []
    },
    "output_all": {
      "person": "Peter Høeg",
      "qid": "Q337384",
      "birthday": "1957-05-17",
      "students": []
    }
  },
  {
    "input": {
      "person": "Karen Blixen",
      "context": "Karen Blixen wrote Out of Africa."
    },
    "output_birthday": {
      "person": "Karen Blixen",
      "qid": "Q182804",
      "birthday": "1885-04-17"
    },
    "output_students": {
      "person": "Karen Blixen",
      "qid": "Q182804",
      "students": []
    },
    "output_all": {
      "person": "Karen Blixen",
      "qid": "Q182804",
      "birthday": "1885-04-17",
      "students": []
    }
  },
  {
    "input": {
      "person": "Victor Borge",
      "context": "Victor Borge was a famous entertainer."
    },
    "output_birthday": {
      "person": "Victor Borge",
      "qid": "Q7925742",
      "birthday": "1965-12-18"
    },
    "output_students": {
      "person": "Victor Borge",
      "qid": "Q7925742",
      "students": []
    },
    "output_all": {
      "person": "Victor Borge",
      "qid": "Q7925742",
      "birthday": "1965-12-18",
      "students": []
    }
  },
  {
    "input": {
      "person": "Ørsted",
      "context": "Ørsted discovered electromagnetism."
    },
    "output_birthday": {
      "person": "Hans Christian Ørsted",
      "qid": "Q44412",
      "birthday": "1777-08-14"
    },
    "output_students": {
      "person": "Hans Christian Ørsted",
      "qid": "Q44412",
      "students": [
        {
          "label": "Carl Holten",
          "qid": "Q15139289"
        },
        {
          "label": "Christopher Hansteen",
          "qid": "Q705048"
        }
      ]
    },
    "output_all": {
      "person": "Hans Christian Ørsted",
      "qid": "Q44412",
      "birthday": "1777-08-14",
      "students": [
        {
          "label": "Carl Holten",
          "qid": "Q15139289"
        },
        {
          "label": "Christopher Hansteen",
          "qid": "Q705048"
        }
      ]
    }
  },
  {
    "input": {
      "person": "Ingrid Bergman",
      "context": "Ingrid Bergman starred in Casablanca."
    },
    "output_birthday": {
      "person": "Ingrid Bergman",
      "qid": "Q43247",
      "birthday": "1915-08-29"
    },
    "output_students": {
      "person": "Ingrid Bergman",
      "qid": "Q43247",
      "students": []
    },
    "output_all": {
      "person": "Ingrid Bergman",
      "qid": "Q43247",
      "birthday": "1915-08-29",
      "students": []
    }
  },
  {
    "input": {
      "person": "Kasper Schmeichel",
      "context": "Kasper Schmeichel played as goalkeeper."
    },
    "output_birthday": {
      "person": "Kasper Schmeichel",
      "qid": "Q295797",
      "birthday": "1986-11-05"
    },
    "output_students": {
      "person": "Kasper Schmeichel",
      "qid": "Q295797",
      "students": []
    },
    "output_all": {
      "person": "Kasper Schmeichel",
      "qid": "Q295797",
      "birthday": "1986-11-05",
      "students": []
    }
  },
  {
    "input": {
      "person": "Greta Thunberg",
      "context": "Greta Thunberg is a climate activist."
    },
    "output_birthday": {
      "person": "Greta Thunberg",
      "qid": "Q56434717",
      "birthday": "2003-01-03"
    },
    "output_students": {
      "person": "Greta Thunberg",
      "qid": "Q56434717",
      "students": []
    },
    "output_all": {
      "person": "Greta Thunberg",
      "qid": "Q56434717",
      "birthday": "2003-01-03",
      "students": []
    }
  },
  {
    "input": {
      "person": "Barack Obama",
      "context": "Barack Obama was president of the United States."
    },
    "output_birthday": {
      "person": "Barack Obama",
      "qid": "Q76",
      "birthday": "1961-08-04"
    },
    "output_students": {
      "person": "Barack Obama",
      "qid": "Q76",
      "students": []
    },
    "output_all": {
      "person": "Barack Obama",
      "qid": "Q76",
      "birthday": "1961-08-04",
      "students": []
    }
  }
]


# ---------------------------------------------------------------------
# Scoring Logic
# ---------------------------------------------------------------------

def compare_outputs(expected, actual):
    """
    Compare expected and actual JSON response.

    Returns
    -------
    tuple
        (is_correct, message)
    """
    if not isinstance(actual, dict):
        return False, "Response is not valid JSON."

    for key in expected:
        if key not in actual:
            return False, f"Missing field: {key}"

    # Exact match comparison (simple but pedagogical)
    if expected == actual:
        return True, "Correct."

    return False, ("Fields differ from expected output. "
                   "Expected: " + json.dumps(expected),
                   " Actual: " + json.dumps(actual))


# ---------------------------------------------------------------------
# Frontend Page
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
<title>DTU Entity Linking Test Interface</title>
</head>
<body style="font-family: Arial; margin: 40px; background-color: white; color: black;">

<h1 style="color: rgb(153,0,0);">DTU Entity Linking Automatic Tester</h1>

<p>
This interface tests the independent entity linking Web service and
automatically scores correctness against a gold dataset.
</p>

<label>Endpoint:</label>
<select id="endpoint">
  <option value="/v1/birthday">/v1/birthday</option>
  <option value="/v1/students">/v1/students</option>
  <option value="/v1/all" selected>/v1/all</option>
</select>

<br><br>

<label>Number of persons:</label>
<select id="sample_size">
  <option value="3" selected>3 (default)</option>
  <option value="20">All 20</option>
</select>

<br><br>

<button onclick="runTest()" 
style="background-color: rgb(153,0,0); color: white; padding: 10px; border: none;">
Run Automatic Test
</button>

<h2>Operational Metrics</h2>
<div id="metrics"></div>

<h2>Detailed Results</h2>
<pre id="results" style="background-color: #f5f5f5; padding: 15px;"></pre>

<script>

async function runTest() {

    document.getElementById("results").textContent = "Running test...";
    document.getElementById("metrics").textContent = "";

    const endpoint = document.getElementById("endpoint").value;
    const sampleSize = parseInt(document.getElementById("sample_size").value);

    const response = await fetch("/run_test", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            endpoint: endpoint,
            sample_size: sampleSize
        })
    });

    const data = await response.json();

    document.getElementById("metrics").innerHTML =
        "<b>Total:</b> " + data.total +
        "<br><b>Correct:</b> " + data.correct +
        "<br><b>Incorrect:</b> " + data.incorrect +
        "<br><b>Average latency (ms):</b> " + data.avg_latency.toFixed(2) +
        "<br><b>Total runtime (ms):</b> " + data.total_runtime.toFixed(2);

    document.getElementById("results").textContent =
        JSON.stringify(data.details, null, 2);
}

</script>

</body>
</html>
"""


# ---------------------------------------------------------------------
# Asynchronous Test Execution
# ---------------------------------------------------------------------

@app.post("/run_test")
async def run_test(request: Request):

    body = await request.json()
    endpoint = body["endpoint"]
    sample_size = body["sample_size"]

    if sample_size > len(DATASET):
        sample_size = len(DATASET)

    sample = random.sample(DATASET, sample_size)

    start_total = time.perf_counter()

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:

        async def test_person(entry):

            expected = entry["output_" + endpoint.replace("/v1/", "")]
            payload = entry["input"]

            start = time.perf_counter()

            try:
                response = await client.post(
                    SERVICE_BASE_URL + endpoint,
                    json=payload
                )

                latency = (time.perf_counter() - start) * 1000

                if response.status_code != 200:
                    return {
                        "person": payload["person"],
                        "status": "HTTP error",
                        "code": response.status_code,
                        "latency_ms": latency
                    }

                actual = response.json()
                correct, message = compare_outputs(expected, actual)

                return {
                    "person": payload["person"],
                    "status": "Correct" if correct else "Incorrect",
                    "message": message,
                    "latency_ms": latency
                }

            except httpx.TimeoutException:
                return {
                    "person": payload["person"],
                    "status": "Timeout",
                    "message": "Service did not respond within timeout.",
                    "latency_ms": TIMEOUT_SECONDS * 1000
                }

            except Exception as e:
                return {
                    "person": payload["person"],
                    "status": "Error",
                    "message": str(e)
                }

        tasks = [test_person(entry) for entry in sample]
        results = await asyncio.gather(*tasks)

    total_runtime = (time.perf_counter() - start_total) * 1000

    correct = sum(1 for r in results if r["status"] == "Correct")
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return JSONResponse({
        "total": sample_size,
        "correct": correct,
        "incorrect": sample_size - correct,
        "avg_latency": avg_latency,
        "total_runtime": total_runtime,
        "details": results
    })
