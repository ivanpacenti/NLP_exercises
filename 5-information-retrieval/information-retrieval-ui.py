"""
Frontend Web application for testing a DTU course information retrieval web service.

This frontend communicates with another independently developed web service
implementing endpoints such as:

    GET /v1/search
    GET /v1/courses/{course_id}/similar
    GET /v1/objectives/search
    GET /v1/health

The goal is to make it easy for students and teachers to:

- Demonstrate their IR system
- Manually test queries
- Automatically evaluate a small test dataset
- Observe latency and service behaviour

The application intentionally has few dependencies and simple code so that
students can easily understand it.

Dependencies
------------
fastapi
uvicorn
httpx

Run
---
uvicorn information-retrieval-app:app --reload --port 8001

Then open:

http://localhost:8001

"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IR_SERVICE = "http://localhost:8000"

REQUEST_TIMEOUT = 5.0

# Paste evaluation dataset here
TEST_DATA: List[Dict[str, Any]] = [
  {
    "id": "teacher_bjorn_full",
    "query": "Bjørn Sand Jensen",
    "intent": "teacher",
    "notes": "Full name with diacritics",
    "relevant": [
      {
        "course_id": "02451",
        "relevance": 2
      },
      {
        "course_id": "02452",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_bjorn_ascii",
    "query": "bjorn sand jensen",
    "intent": "robustness",
    "notes": "ASCII normalized teacher name",
    "relevant": [
      {
        "course_id": "02451",
        "relevance": 2
      },
      {
        "course_id": "02452",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_bjorn_permuted",
    "query": "jensen bjørn sand",
    "intent": "teacher",
    "notes": "Permuted parts",
    "relevant": [
      {
        "course_id": "02451",
        "relevance": 2
      },
      {
        "course_id": "02452",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_bjorn_add_word",
    "query": "teacher bjørn sand jensen",
    "intent": "teacher",
    "notes": "Add irrelevant word to name",
    "relevant": [
      {
        "course_id": "02451",
        "relevance": 2
      },
      {
        "course_id": "02452",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_ivana_full",
    "query": "Ivana Konvalinka",
    "intent": "teacher",
    "notes": "Full teacher name",
    "relevant": [
      {
        "course_id": "02455",
        "relevance": 2
      },
      {
        "course_id": "02464",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_ivana_first",
    "query": "ivana",
    "intent": "teacher",
    "notes": "First name only",
    "relevant": [
      {
        "course_id": "02455",
        "relevance": 2
      },
      {
        "course_id": "02464",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_ivana_last",
    "query": "konvalinka",
    "intent": "teacher",
    "notes": "Lastname query",
    "relevant": [
      {
        "course_id": "02455",
        "relevance": 2
      },
      {
        "course_id": "02464",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_christian_kim_christiansen",
    "query": "Christian Kim Christiansen",
    "intent": "teacher",
    "notes": "Teacher name as in source; appears in 17 courses.",
    "relevant": [
      {
        "course_id": "41861",
        "relevance": 2
      },
      {
        "course_id": "62010",
        "relevance": 2
      },
      {
        "course_id": "62143",
        "relevance": 2
      },
      {
        "course_id": "62207",
        "relevance": 2
      },
      {
        "course_id": "62602",
        "relevance": 2
      },
      {
        "course_id": "62646",
        "relevance": 2
      },
      {
        "course_id": "62677",
        "relevance": 2
      },
      {
        "course_id": "62683",
        "relevance": 2
      },
      {
        "course_id": "62696",
        "relevance": 2
      },
      {
        "course_id": "62801",
        "relevance": 2
      },
      {
        "course_id": "62802",
        "relevance": 2
      },
      {
        "course_id": "62804",
        "relevance": 2
      },
      {
        "course_id": "62805",
        "relevance": 2
      },
      {
        "course_id": "62806",
        "relevance": 2
      },
      {
        "course_id": "62809",
        "relevance": 2
      },
      {
        "course_id": "62811",
        "relevance": 2
      },
      {
        "course_id": "MA143",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_kaj_age_henneberg",
    "query": "Kaj-Åge Henneberg",
    "intent": "teacher",
    "notes": "Teacher name as in source; appears in 13 courses.",
    "relevant": [
      {
        "course_id": "22050",
        "relevance": 2
      },
      {
        "course_id": "22439",
        "relevance": 2
      },
      {
        "course_id": "22460",
        "relevance": 2
      },
      {
        "course_id": "22461",
        "relevance": 2
      },
      {
        "course_id": "22462",
        "relevance": 2
      },
      {
        "course_id": "KU002",
        "relevance": 2
      },
      {
        "course_id": "KU003",
        "relevance": 2
      },
      {
        "course_id": "KU004",
        "relevance": 2
      },
      {
        "course_id": "KU005",
        "relevance": 2
      },
      {
        "course_id": "KU006",
        "relevance": 2
      },
      {
        "course_id": "KU010",
        "relevance": 2
      },
      {
        "course_id": "KU011",
        "relevance": 2
      },
      {
        "course_id": "KU013",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_kaj_age_henneberg_ascii",
    "query": "Kaj-Age Henneberg",
    "intent": "robustness",
    "notes": "Teacher name with diacritics stripped (ASCII variant).",
    "relevant": [
      {
        "course_id": "22050",
        "relevance": 2
      },
      {
        "course_id": "22439",
        "relevance": 2
      },
      {
        "course_id": "22460",
        "relevance": 2
      },
      {
        "course_id": "22461",
        "relevance": 2
      },
      {
        "course_id": "22462",
        "relevance": 2
      },
      {
        "course_id": "KU002",
        "relevance": 2
      },
      {
        "course_id": "KU003",
        "relevance": 2
      },
      {
        "course_id": "KU004",
        "relevance": 2
      },
      {
        "course_id": "KU005",
        "relevance": 2
      },
      {
        "course_id": "KU006",
        "relevance": 2
      },
      {
        "course_id": "KU010",
        "relevance": 2
      },
      {
        "course_id": "KU011",
        "relevance": 2
      },
      {
        "course_id": "KU013",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_bo_ea_holst_christensen",
    "query": "Bo Ea Holst-Christensen",
    "intent": "teacher",
    "notes": "Teacher name as in source; appears in 12 courses.",
    "relevant": [
      {
        "course_id": "62229",
        "relevance": 2
      },
      {
        "course_id": "62416",
        "relevance": 2
      },
      {
        "course_id": "62417",
        "relevance": 2
      },
      {
        "course_id": "62424",
        "relevance": 2
      },
      {
        "course_id": "62444",
        "relevance": 2
      },
      {
        "course_id": "62450",
        "relevance": 2
      },
      {
        "course_id": "62453",
        "relevance": 2
      },
      {
        "course_id": "62484",
        "relevance": 2
      },
      {
        "course_id": "62512",
        "relevance": 2
      },
      {
        "course_id": "62521",
        "relevance": 2
      },
      {
        "course_id": "62531",
        "relevance": 2
      },
      {
        "course_id": "62533",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_christian_danvad_damsgaard",
    "query": "Christian Danvad Damsgaard",
    "intent": "teacher",
    "notes": "Teacher name as in source; appears in 12 courses.",
    "relevant": [
      {
        "course_id": "10240",
        "relevance": 2
      },
      {
        "course_id": "10331",
        "relevance": 2
      },
      {
        "course_id": "10509",
        "relevance": 2
      },
      {
        "course_id": "10720",
        "relevance": 2
      },
      {
        "course_id": "10721",
        "relevance": 2
      },
      {
        "course_id": "10722",
        "relevance": 2
      },
      {
        "course_id": "10730",
        "relevance": 2
      },
      {
        "course_id": "10731",
        "relevance": 2
      },
      {
        "course_id": "10732",
        "relevance": 2
      },
      {
        "course_id": "22602",
        "relevance": 2
      },
      {
        "course_id": "34019",
        "relevance": 2
      },
      {
        "course_id": "34029",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_harry_bradford_bingham",
    "query": "Harry Bradford Bingham",
    "intent": "teacher",
    "notes": "Teacher name as in source; appears in 11 courses.",
    "relevant": [
      {
        "course_id": "41201",
        "relevance": 2
      },
      {
        "course_id": "41216",
        "relevance": 2
      },
      {
        "course_id": "41222",
        "relevance": 2
      },
      {
        "course_id": "41263",
        "relevance": 2
      },
      {
        "course_id": "41270",
        "relevance": 2
      },
      {
        "course_id": "41271",
        "relevance": 2
      },
      {
        "course_id": "41275",
        "relevance": 2
      },
      {
        "course_id": "41292",
        "relevance": 2
      },
      {
        "course_id": "41317",
        "relevance": 2
      },
      {
        "course_id": "41341",
        "relevance": 2
      },
      {
        "course_id": "41342",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_line_juul_larsen",
    "query": "Line Juul Larsen",
    "intent": "teacher",
    "notes": "Teacher name as in source; appears in 11 courses.",
    "relevant": [
      {
        "course_id": "KU002",
        "relevance": 2
      },
      {
        "course_id": "KU005",
        "relevance": 2
      },
      {
        "course_id": "KU010",
        "relevance": 2
      },
      {
        "course_id": "KU011",
        "relevance": 2
      },
      {
        "course_id": "KU013",
        "relevance": 2
      },
      {
        "course_id": "KU101",
        "relevance": 2
      },
      {
        "course_id": "KU103",
        "relevance": 2
      },
      {
        "course_id": "KU105",
        "relevance": 2
      },
      {
        "course_id": "KU112",
        "relevance": 2
      },
      {
        "course_id": "KU115",
        "relevance": 2
      },
      {
        "course_id": "KU181",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_martin_dufva",
    "query": "Martin Dufva",
    "intent": "teacher",
    "notes": "Teacher name as in source; appears in 11 courses.",
    "relevant": [
      {
        "course_id": "22205",
        "relevance": 2
      },
      {
        "course_id": "22206",
        "relevance": 2
      },
      {
        "course_id": "22207",
        "relevance": 2
      },
      {
        "course_id": "22212",
        "relevance": 2
      },
      {
        "course_id": "22213",
        "relevance": 2
      },
      {
        "course_id": "22283",
        "relevance": 2
      },
      {
        "course_id": "22285",
        "relevance": 2
      },
      {
        "course_id": "25106",
        "relevance": 2
      },
      {
        "course_id": "KU112",
        "relevance": 2
      },
      {
        "course_id": "KU113",
        "relevance": 2
      },
      {
        "course_id": "KU115",
        "relevance": 2
      }
    ]
  },
  {
    "id": "teacher_jerome_chenevez",
    "query": "Jerome Chenevez",
    "intent": "teacher",
    "notes": "Teacher name as in source; appears in 10 courses.",
    "relevant": [
      {
        "course_id": "30102",
        "relevance": 2
      },
      {
        "course_id": "30110",
        "relevance": 2
      },
      {
        "course_id": "30121",
        "relevance": 2
      },
      {
        "course_id": "30220",
        "relevance": 2
      },
      {
        "course_id": "30221",
        "relevance": 2
      },
      {
        "course_id": "30222",
        "relevance": 2
      },
      {
        "course_id": "30223",
        "relevance": 2
      },
      {
        "course_id": "30224",
        "relevance": 2
      },
      {
        "course_id": "30791",
        "relevance": 2
      },
      {
        "course_id": "30794",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_research_immersion",
    "query": "research immersion",
    "intent": "topic",
    "notes": "Phrase from titles; appears in ~10 titles; should retrieve these courses.",
    "relevant": [
      {
        "course_id": "10720",
        "relevance": 2
      },
      {
        "course_id": "10721",
        "relevance": 2
      },
      {
        "course_id": "10722",
        "relevance": 2
      },
      {
        "course_id": "10730",
        "relevance": 2
      },
      {
        "course_id": "10731",
        "relevance": 2
      },
      {
        "course_id": "10732",
        "relevance": 2
      },
      {
        "course_id": "22107",
        "relevance": 2
      },
      {
        "course_id": "22611",
        "relevance": 2
      },
      {
        "course_id": "22612",
        "relevance": 2
      },
      {
        "course_id": "34658",
        "relevance": 2
      },
      {
        "course_id": "34659",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_research_immersion_upper",
    "query": "RESEARCH IMMERSION",
    "intent": "robustness",
    "notes": "Upper-case variant to test casing robustness.",
    "relevant": [
      {
        "course_id": "10720",
        "relevance": 2
      },
      {
        "course_id": "10721",
        "relevance": 2
      },
      {
        "course_id": "10722",
        "relevance": 2
      },
      {
        "course_id": "10730",
        "relevance": 2
      },
      {
        "course_id": "10731",
        "relevance": 2
      },
      {
        "course_id": "10732",
        "relevance": 2
      },
      {
        "course_id": "22107",
        "relevance": 2
      },
      {
        "course_id": "22611",
        "relevance": 2
      },
      {
        "course_id": "22612",
        "relevance": 2
      },
      {
        "course_id": "34658",
        "relevance": 2
      },
      {
        "course_id": "34659",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_research_immersion_typo",
    "query": "research immesion",
    "intent": "robustness",
    "notes": "Single small typo variant to test typo tolerance.",
    "relevant": [
      {
        "course_id": "10720",
        "relevance": 2
      },
      {
        "course_id": "10721",
        "relevance": 2
      },
      {
        "course_id": "10722",
        "relevance": 2
      },
      {
        "course_id": "10730",
        "relevance": 2
      },
      {
        "course_id": "10731",
        "relevance": 2
      },
      {
        "course_id": "10732",
        "relevance": 2
      },
      {
        "course_id": "22107",
        "relevance": 2
      },
      {
        "course_id": "22611",
        "relevance": 2
      },
      {
        "course_id": "22612",
        "relevance": 2
      },
      {
        "course_id": "34658",
        "relevance": 2
      },
      {
        "course_id": "34659",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_space_physics",
    "query": "space physics",
    "intent": "topic",
    "notes": "Phrase from titles; appears in ~10 titles; should retrieve these courses.",
    "relevant": [
      {
        "course_id": "30110",
        "relevance": 2
      },
      {
        "course_id": "30202",
        "relevance": 2
      },
      {
        "course_id": "30220",
        "relevance": 2
      },
      {
        "course_id": "30221",
        "relevance": 2
      },
      {
        "course_id": "30222",
        "relevance": 2
      },
      {
        "course_id": "30223",
        "relevance": 2
      },
      {
        "course_id": "30224",
        "relevance": 2
      },
      {
        "course_id": "30720",
        "relevance": 2
      },
      {
        "course_id": "30760",
        "relevance": 2
      },
      {
        "course_id": "30911",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_space_physics_upper",
    "query": "SPACE PHYSICS",
    "intent": "robustness",
    "notes": "Upper-case variant to test casing robustness.",
    "relevant": [
      {
        "course_id": "30110",
        "relevance": 2
      },
      {
        "course_id": "30202",
        "relevance": 2
      },
      {
        "course_id": "30220",
        "relevance": 2
      },
      {
        "course_id": "30221",
        "relevance": 2
      },
      {
        "course_id": "30222",
        "relevance": 2
      },
      {
        "course_id": "30223",
        "relevance": 2
      },
      {
        "course_id": "30224",
        "relevance": 2
      },
      {
        "course_id": "30720",
        "relevance": 2
      },
      {
        "course_id": "30760",
        "relevance": 2
      },
      {
        "course_id": "30911",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_space_physics_typo",
    "query": "space phyics",
    "intent": "robustness",
    "notes": "Single small typo variant to test typo tolerance.",
    "relevant": [
      {
        "course_id": "30110",
        "relevance": 2
      },
      {
        "course_id": "30202",
        "relevance": 2
      },
      {
        "course_id": "30220",
        "relevance": 2
      },
      {
        "course_id": "30221",
        "relevance": 2
      },
      {
        "course_id": "30222",
        "relevance": 2
      },
      {
        "course_id": "30223",
        "relevance": 2
      },
      {
        "course_id": "30224",
        "relevance": 2
      },
      {
        "course_id": "30720",
        "relevance": 2
      },
      {
        "course_id": "30760",
        "relevance": 2
      },
      {
        "course_id": "30911",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_artificial_intelligence",
    "query": "artificial intelligence",
    "intent": "topic",
    "notes": "Phrase from titles; appears in ~9 titles; should retrieve these courses.",
    "relevant": [
      {
        "course_id": "02180",
        "relevance": 2
      },
      {
        "course_id": "02182",
        "relevance": 2
      },
      {
        "course_id": "02285",
        "relevance": 2
      },
      {
        "course_id": "02445",
        "relevance": 2
      },
      {
        "course_id": "02464",
        "relevance": 2
      },
      {
        "course_id": "02466",
        "relevance": 2
      },
      {
        "course_id": "10316",
        "relevance": 2
      },
      {
        "course_id": "23212",
        "relevance": 2
      },
      {
        "course_id": "23564",
        "relevance": 2
      },
      {
        "course_id": "27003",
        "relevance": 2
      },
      {
        "course_id": "27666",
        "relevance": 2
      },
      {
        "course_id": "47341",
        "relevance": 2
      },
      {
        "course_id": "62200",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_artificial_intelligence_upper",
    "query": "ARTIFICIAL INTELLIGENCE",
    "intent": "robustness",
    "notes": "Upper-case variant to test casing robustness.",
    "relevant": [
      {
        "course_id": "02180",
        "relevance": 2
      },
      {
        "course_id": "02182",
        "relevance": 2
      },
      {
        "course_id": "02285",
        "relevance": 2
      },
      {
        "course_id": "02445",
        "relevance": 2
      },
      {
        "course_id": "02464",
        "relevance": 2
      },
      {
        "course_id": "02466",
        "relevance": 2
      },
      {
        "course_id": "10316",
        "relevance": 2
      },
      {
        "course_id": "23212",
        "relevance": 2
      },
      {
        "course_id": "23564",
        "relevance": 2
      },
      {
        "course_id": "27003",
        "relevance": 2
      },
      {
        "course_id": "27666",
        "relevance": 2
      },
      {
        "course_id": "47341",
        "relevance": 2
      },
      {
        "course_id": "62200",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_artificial_intelligence_typo",
    "query": "artificial intellgence",
    "intent": "robustness",
    "notes": "Single small typo variant to test typo tolerance.",
    "relevant": [
      {
        "course_id": "02180",
        "relevance": 2
      },
      {
        "course_id": "02182",
        "relevance": 2
      },
      {
        "course_id": "02285",
        "relevance": 2
      },
      {
        "course_id": "02445",
        "relevance": 2
      },
      {
        "course_id": "02464",
        "relevance": 2
      },
      {
        "course_id": "02466",
        "relevance": 2
      },
      {
        "course_id": "10316",
        "relevance": 2
      },
      {
        "course_id": "23212",
        "relevance": 2
      },
      {
        "course_id": "23564",
        "relevance": 2
      },
      {
        "course_id": "27003",
        "relevance": 2
      },
      {
        "course_id": "27666",
        "relevance": 2
      },
      {
        "course_id": "47341",
        "relevance": 2
      },
      {
        "course_id": "62200",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_chemical_engineering",
    "query": "chemical engineering",
    "intent": "topic",
    "notes": "Phrase from titles; appears in ~9 titles; should retrieve these courses.",
    "relevant": [
      {
        "course_id": "26010",
        "relevance": 2
      },
      {
        "course_id": "28010",
        "relevance": 2
      },
      {
        "course_id": "28012",
        "relevance": 2
      },
      {
        "course_id": "28016",
        "relevance": 2
      },
      {
        "course_id": "28020",
        "relevance": 2
      },
      {
        "course_id": "28021",
        "relevance": 2
      },
      {
        "course_id": "28022",
        "relevance": 2
      },
      {
        "course_id": "28025",
        "relevance": 2
      },
      {
        "course_id": "28121",
        "relevance": 2
      },
      {
        "course_id": "28123",
        "relevance": 2
      },
      {
        "course_id": "28125",
        "relevance": 2
      },
      {
        "course_id": "28140",
        "relevance": 2
      },
      {
        "course_id": "28145",
        "relevance": 2
      },
      {
        "course_id": "28150",
        "relevance": 2
      },
      {
        "course_id": "28155",
        "relevance": 2
      },
      {
        "course_id": "28157",
        "relevance": 2
      },
      {
        "course_id": "28160",
        "relevance": 2
      },
      {
        "course_id": "28165",
        "relevance": 2
      },
      {
        "course_id": "28212",
        "relevance": 2
      },
      {
        "course_id": "28213",
        "relevance": 2
      },
      {
        "course_id": "28214",
        "relevance": 2
      },
      {
        "course_id": "28216",
        "relevance": 2
      },
      {
        "course_id": "28217",
        "relevance": 2
      },
      {
        "course_id": "28221",
        "relevance": 2
      },
      {
        "course_id": "28233",
        "relevance": 2
      },
      {
        "course_id": "28242",
        "relevance": 2
      },
      {
        "course_id": "28244",
        "relevance": 2
      },
      {
        "course_id": "28271",
        "relevance": 2
      },
      {
        "course_id": "28311",
        "relevance": 2
      },
      {
        "course_id": "28315",
        "relevance": 2
      },
      {
        "course_id": "28316",
        "relevance": 2
      },
      {
        "course_id": "28322",
        "relevance": 2
      },
      {
        "course_id": "28342",
        "relevance": 2
      },
      {
        "course_id": "28344",
        "relevance": 2
      },
      {
        "course_id": "28345",
        "relevance": 2
      },
      {
        "course_id": "28346",
        "relevance": 2
      },
      {
        "course_id": "28350",
        "relevance": 2
      },
      {
        "course_id": "28352",
        "relevance": 2
      },
      {
        "course_id": "28361",
        "relevance": 2
      },
      {
        "course_id": "28412",
        "relevance": 2
      },
      {
        "course_id": "28420",
        "relevance": 2
      },
      {
        "course_id": "28423",
        "relevance": 2
      },
      {
        "course_id": "28434",
        "relevance": 2
      },
      {
        "course_id": "28443",
        "relevance": 2
      },
      {
        "course_id": "28451",
        "relevance": 2
      },
      {
        "course_id": "28455",
        "relevance": 2
      },
      {
        "course_id": "28485",
        "relevance": 2
      },
      {
        "course_id": "28530",
        "relevance": 2
      },
      {
        "course_id": "28720",
        "relevance": 2
      },
      {
        "course_id": "28725",
        "relevance": 2
      },
      {
        "course_id": "28730",
        "relevance": 2
      },
      {
        "course_id": "28737",
        "relevance": 2
      },
      {
        "course_id": "28745",
        "relevance": 2
      },
      {
        "course_id": "28747",
        "relevance": 2
      },
      {
        "course_id": "28750",
        "relevance": 2
      },
      {
        "course_id": "28751",
        "relevance": 2
      },
      {
        "course_id": "28755",
        "relevance": 2
      },
      {
        "course_id": "28761",
        "relevance": 2
      },
      {
        "course_id": "28811",
        "relevance": 2
      },
      {
        "course_id": "28831",
        "relevance": 2
      },
      {
        "course_id": "28845",
        "relevance": 2
      },
      {
        "course_id": "28850",
        "relevance": 2
      },
      {
        "course_id": "28852",
        "relevance": 2
      },
      {
        "course_id": "28855",
        "relevance": 2
      },
      {
        "course_id": "28857",
        "relevance": 2
      },
      {
        "course_id": "28870",
        "relevance": 2
      },
      {
        "course_id": "28871",
        "relevance": 2
      },
      {
        "course_id": "28872",
        "relevance": 2
      },
      {
        "course_id": "28905",
        "relevance": 2
      },
      {
        "course_id": "28908",
        "relevance": 2
      },
      {
        "course_id": "28909",
        "relevance": 2
      },
      {
        "course_id": "28917",
        "relevance": 2
      },
      {
        "course_id": "28923",
        "relevance": 2
      },
      {
        "course_id": "28927",
        "relevance": 2
      },
      {
        "course_id": "28928",
        "relevance": 2
      },
      {
        "course_id": "28930",
        "relevance": 2
      },
      {
        "course_id": "28932",
        "relevance": 2
      },
      {
        "course_id": "28934",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_chemical_engineering_upper",
    "query": "CHEMICAL ENGINEERING",
    "intent": "robustness",
    "notes": "Upper-case variant to test casing robustness.",
    "relevant": [
      {
        "course_id": "26010",
        "relevance": 2
      },
      {
        "course_id": "28010",
        "relevance": 2
      },
      {
        "course_id": "28012",
        "relevance": 2
      },
      {
        "course_id": "28016",
        "relevance": 2
      },
      {
        "course_id": "28020",
        "relevance": 2
      },
      {
        "course_id": "28021",
        "relevance": 2
      },
      {
        "course_id": "28022",
        "relevance": 2
      },
      {
        "course_id": "28025",
        "relevance": 2
      },
      {
        "course_id": "28121",
        "relevance": 2
      },
      {
        "course_id": "28123",
        "relevance": 2
      },
      {
        "course_id": "28125",
        "relevance": 2
      },
      {
        "course_id": "28140",
        "relevance": 2
      },
      {
        "course_id": "28145",
        "relevance": 2
      },
      {
        "course_id": "28150",
        "relevance": 2
      },
      {
        "course_id": "28155",
        "relevance": 2
      },
      {
        "course_id": "28157",
        "relevance": 2
      },
      {
        "course_id": "28160",
        "relevance": 2
      },
      {
        "course_id": "28165",
        "relevance": 2
      },
      {
        "course_id": "28212",
        "relevance": 2
      },
      {
        "course_id": "28213",
        "relevance": 2
      },
      {
        "course_id": "28214",
        "relevance": 2
      },
      {
        "course_id": "28216",
        "relevance": 2
      },
      {
        "course_id": "28217",
        "relevance": 2
      },
      {
        "course_id": "28221",
        "relevance": 2
      },
      {
        "course_id": "28233",
        "relevance": 2
      },
      {
        "course_id": "28242",
        "relevance": 2
      },
      {
        "course_id": "28244",
        "relevance": 2
      },
      {
        "course_id": "28271",
        "relevance": 2
      },
      {
        "course_id": "28311",
        "relevance": 2
      },
      {
        "course_id": "28315",
        "relevance": 2
      },
      {
        "course_id": "28316",
        "relevance": 2
      },
      {
        "course_id": "28322",
        "relevance": 2
      },
      {
        "course_id": "28342",
        "relevance": 2
      },
      {
        "course_id": "28344",
        "relevance": 2
      },
      {
        "course_id": "28345",
        "relevance": 2
      },
      {
        "course_id": "28346",
        "relevance": 2
      },
      {
        "course_id": "28350",
        "relevance": 2
      },
      {
        "course_id": "28352",
        "relevance": 2
      },
      {
        "course_id": "28361",
        "relevance": 2
      },
      {
        "course_id": "28412",
        "relevance": 2
      },
      {
        "course_id": "28420",
        "relevance": 2
      },
      {
        "course_id": "28423",
        "relevance": 2
      },
      {
        "course_id": "28434",
        "relevance": 2
      },
      {
        "course_id": "28443",
        "relevance": 2
      },
      {
        "course_id": "28451",
        "relevance": 2
      },
      {
        "course_id": "28455",
        "relevance": 2
      },
      {
        "course_id": "28485",
        "relevance": 2
      },
      {
        "course_id": "28530",
        "relevance": 2
      },
      {
        "course_id": "28720",
        "relevance": 2
      },
      {
        "course_id": "28725",
        "relevance": 2
      },
      {
        "course_id": "28730",
        "relevance": 2
      },
      {
        "course_id": "28737",
        "relevance": 2
      },
      {
        "course_id": "28745",
        "relevance": 2
      },
      {
        "course_id": "28747",
        "relevance": 2
      },
      {
        "course_id": "28750",
        "relevance": 2
      },
      {
        "course_id": "28751",
        "relevance": 2
      },
      {
        "course_id": "28755",
        "relevance": 2
      },
      {
        "course_id": "28761",
        "relevance": 2
      },
      {
        "course_id": "28811",
        "relevance": 2
      },
      {
        "course_id": "28831",
        "relevance": 2
      },
      {
        "course_id": "28845",
        "relevance": 2
      },
      {
        "course_id": "28850",
        "relevance": 2
      },
      {
        "course_id": "28852",
        "relevance": 2
      },
      {
        "course_id": "28855",
        "relevance": 2
      },
      {
        "course_id": "28857",
        "relevance": 2
      },
      {
        "course_id": "28870",
        "relevance": 2
      },
      {
        "course_id": "28871",
        "relevance": 2
      },
      {
        "course_id": "28872",
        "relevance": 2
      },
      {
        "course_id": "28905",
        "relevance": 2
      },
      {
        "course_id": "28908",
        "relevance": 2
      },
      {
        "course_id": "28909",
        "relevance": 2
      },
      {
        "course_id": "28917",
        "relevance": 2
      },
      {
        "course_id": "28923",
        "relevance": 2
      },
      {
        "course_id": "28927",
        "relevance": 2
      },
      {
        "course_id": "28928",
        "relevance": 2
      },
      {
        "course_id": "28930",
        "relevance": 2
      },
      {
        "course_id": "28932",
        "relevance": 2
      },
      {
        "course_id": "28934",
        "relevance": 2
      }
    ]
  },
  {
    "id": "topic_chemical_engineering_typo",
    "query": "chemical enginering",
    "intent": "robustness",
    "notes": "Single small typo variant to test typo tolerance.",
    "relevant": [
      {
        "course_id": "26010",
        "relevance": 2
      },
      {
        "course_id": "28010",
        "relevance": 2
      },
      {
        "course_id": "28012",
        "relevance": 2
      },
      {
        "course_id": "28016",
        "relevance": 2
      },
      {
        "course_id": "28020",
        "relevance": 2
      },
      {
        "course_id": "28021",
        "relevance": 2
      },
      {
        "course_id": "28022",
        "relevance": 2
      },
      {
        "course_id": "28025",
        "relevance": 2
      },
      {
        "course_id": "28121",
        "relevance": 2
      },
      {
        "course_id": "28123",
        "relevance": 2
      },
      {
        "course_id": "28125",
        "relevance": 2
      },
      {
        "course_id": "28140",
        "relevance": 2
      },
      {
        "course_id": "28145",
        "relevance": 2
      },
      {
        "course_id": "28150",
        "relevance": 2
      },
      {
        "course_id": "28155",
        "relevance": 2
      },
      {
        "course_id": "28157",
        "relevance": 2
      },
      {
        "course_id": "28160",
        "relevance": 2
      },
      {
        "course_id": "28165",
        "relevance": 2
      },
      {
        "course_id": "28212",
        "relevance": 2
      },
      {
        "course_id": "28213",
        "relevance": 2
      },
      {
        "course_id": "28214",
        "relevance": 2
      },
      {
        "course_id": "28216",
        "relevance": 2
      },
      {
        "course_id": "28217",
        "relevance": 2
      },
      {
        "course_id": "28221",
        "relevance": 2
      },
      {
        "course_id": "28233",
        "relevance": 2
      },
      {
        "course_id": "28242",
        "relevance": 2
      },
      {
        "course_id": "28244",
        "relevance": 2
      },
      {
        "course_id": "28271",
        "relevance": 2
      },
      {
        "course_id": "28311",
        "relevance": 2
      },
      {
        "course_id": "28315",
        "relevance": 2
      },
      {
        "course_id": "28316",
        "relevance": 2
      },
      {
        "course_id": "28322",
        "relevance": 2
      },
      {
        "course_id": "28342",
        "relevance": 2
      },
      {
        "course_id": "28344",
        "relevance": 2
      },
      {
        "course_id": "28345",
        "relevance": 2
      },
      {
        "course_id": "28346",
        "relevance": 2
      },
      {
        "course_id": "28350",
        "relevance": 2
      },
      {
        "course_id": "28352",
        "relevance": 2
      },
      {
        "course_id": "28361",
        "relevance": 2
      },
      {
        "course_id": "28412",
        "relevance": 2
      },
      {
        "course_id": "28420",
        "relevance": 2
      },
      {
        "course_id": "28423",
        "relevance": 2
      },
      {
        "course_id": "28434",
        "relevance": 2
      },
      {
        "course_id": "28443",
        "relevance": 2
      },
      {
        "course_id": "28451",
        "relevance": 2
      },
      {
        "course_id": "28455",
        "relevance": 2
      },
      {
        "course_id": "28485",
        "relevance": 2
      },
      {
        "course_id": "28530",
        "relevance": 2
      },
      {
        "course_id": "28720",
        "relevance": 2
      },
      {
        "course_id": "28725",
        "relevance": 2
      },
      {
        "course_id": "28730",
        "relevance": 2
      },
      {
        "course_id": "28737",
        "relevance": 2
      },
      {
        "course_id": "28745",
        "relevance": 2
      },
      {
        "course_id": "28747",
        "relevance": 2
      },
      {
        "course_id": "28750",
        "relevance": 2
      },
      {
        "course_id": "28751",
        "relevance": 2
      },
      {
        "course_id": "28755",
        "relevance": 2
      },
      {
        "course_id": "28761",
        "relevance": 2
      },
      {
        "course_id": "28811",
        "relevance": 2
      },
      {
        "course_id": "28831",
        "relevance": 2
      },
      {
        "course_id": "28845",
        "relevance": 2
      },
      {
        "course_id": "28850",
        "relevance": 2
      },
      {
        "course_id": "28852",
        "relevance": 2
      },
      {
        "course_id": "28855",
        "relevance": 2
      },
      {
        "course_id": "28857",
        "relevance": 2
      },
      {
        "course_id": "28870",
        "relevance": 2
      },
      {
        "course_id": "28871",
        "relevance": 2
      },
      {
        "course_id": "28872",
        "relevance": 2
      },
      {
        "course_id": "28905",
        "relevance": 2
      },
      {
        "course_id": "28908",
        "relevance": 2
      },
      {
        "course_id": "28909",
        "relevance": 2
      },
      {
        "course_id": "28917",
        "relevance": 2
      },
      {
        "course_id": "28923",
        "relevance": 2
      },
      {
        "course_id": "28927",
        "relevance": 2
      },
      {
        "course_id": "28928",
        "relevance": 2
      },
      {
        "course_id": "28930",
        "relevance": 2
      },
      {
        "course_id": "28932",
        "relevance": 2
      },
      {
        "course_id": "28934",
        "relevance": 2
      }
    ]
  }
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

async def call_service(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call IR service and measure latency.

    Returns
    -------
    dict
        Dictionary with keys:

        - ok : bool
        - latency : float
        - data : JSON response or error
    """
    url = IR_SERVICE + path

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.get(url, params=params)

        latency = time.perf_counter() - start

        return {
            "ok": True,
            "latency": latency,
            "data": r.json()
        }

    except Exception as e:

        latency = time.perf_counter() - start

        return {
            "ok": False,
            "latency": latency,
            "error": str(e)
        }


async def evaluate_dataset(top_k: int = 20) -> Dict[str, Any]:
    """
    Evaluate the IR service using the test dataset.

    Metrics
    -------
    reciprocal_rank
        1 / rank of first relevant result (0 if none found)

    recall_at_k
        fraction of relevant documents retrieved in top_k

    Returns
    -------
    dict
        Evaluation summary and per-query results.
    """

    query_results = []

    reciprocal_ranks = []
    recalls = []
    latencies = []

    for item in TEST_DATA:

        query = item["query"]

        res = await call_service(
            "/v1/search",
            {"query": query, "top_k": top_k}
        )

        if not res["ok"]:

            query_results.append({
                "query": query,
                "status": "error",
                "error": res["error"]
            })

            continue

        latency = res["latency"]
        latencies.append(latency)

        returned = [r["course_id"] for r in res["data"]["results"]]

        relevant = {r["course_id"] for r in item["relevant"]}

        # ------------------------------------------------
        # rank of first relevant result
        # ------------------------------------------------

        rank = None

        for i, cid in enumerate(returned, start=1):
            if cid in relevant:
                rank = i
                break

        if rank is None:
            rr = 0.0
        else:
            rr = 1.0 / rank

        reciprocal_ranks.append(rr)

        # ------------------------------------------------
        # recall@k
        # ------------------------------------------------

        hits = sum(1 for cid in returned if cid in relevant)

        recall = hits / len(relevant)

        recalls.append(recall)

        query_results.append({
            "query": query,
            "latency": latency,
            "rank_first_relevant": rank,
            "reciprocal_rank": rr,
            "recall_at_k": recall,
            "returned": returned[:top_k],
            "relevant": list(relevant)
        })

    # ----------------------------------------------------
    # overall metrics
    # ----------------------------------------------------

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    mean_recall = sum(recalls) / len(recalls) if recalls else 0

    mean_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "summary": {
            "queries": len(query_results),
            "MRR": mrr,
            "mean_recall_at_k": mean_recall,
            "mean_latency": mean_latency
        },
        "queries": query_results
    }    



# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """
    Main user interface page.
    """

    return f"""
<!DOCTYPE html>
<html>
<head>
<title>DTU Information Retrieval Demo</title>

<style>

body {{
font-family: Arial;
margin:40px;
background:white;
color:black;
}}

h1 {{
color: rgb(153,0,0);
}}

button {{
background: rgb(153,0,0);
color:white;
border:none;
padding:8px 14px;
cursor:pointer;
}}

input {{
padding:6px;
margin:5px;
}}

pre {{
background:#f5f5f5;
padding:10px;
}}

</style>

<script>

async function searchCourses() {{

    let q = document.getElementById("query").value
    let url = "/proxy/search?query=" + encodeURIComponent(q)
    let start = performance.now()
    let r = await fetch(url)
    let j = await r.json()
    let latency = performance.now() - start

    document.getElementById("result").textContent =
        JSON.stringify(j, null, 2)

    document.getElementById("latency").textContent =
        "Latency: " + latency.toFixed(3) + " ms"
}}

async function similarCourses() {{

    let cid = document.getElementById("courseid").value
    let url = "/proxy/similar/" + cid
    let start = performance.now()
    let r = await fetch(url)
    let j = await r.json()
    let latency = performance.now() - start

    document.getElementById("result").textContent =
        JSON.stringify(j, null, 2)

    document.getElementById("latency").textContent =
        "Latency: " + latency.toFixed(3) + " ms"
}}

async function objectiveSearch() {{

    let q = document.getElementById("objective_query").value
    let url = "/proxy/objectives?query=" + encodeURIComponent(q)
    let start = performance.now()
    let r = await fetch(url)
    let j = await r.json()
    let latency = performance.now() - start

    document.getElementById("result").textContent =
       JSON.stringify(j, null, 2)

    document.getElementById("latency").textContent =
        "Latency: " + latency.toFixed(3) + " ms"

}}

async function runTests() {{

    let r = await fetch("/evaluate")
    let j = await r.json()

    document.getElementById("testresult").textContent =
        JSON.stringify(j, null, 2)
}}

</script>
</head>

<body>

<h1>DTU Information Retrieval Demo</h1>

<p>
IR service endpoint: <b>{IR_SERVICE}</b>
</p>

<h2>Search courses</h2>

<input id="query" size="40" placeholder="machine learning">
<button onclick="searchCourses()">Search</button>

<h2>Similar courses</h2>

<input id="courseid" placeholder="02451">
<button onclick="similarCourses()">Find similar</button>

<h2>Search learning objectives</h2>

<input id="objective_query" size="40" placeholder="principal component">
<button onclick="objectiveSearch()">Search</button>

<h2>Latency</h2>

<div id="latency"></div>

<h2>Results</h2>

<pre id="result"></pre>

<h2>Evaluation dataset</h2>

<button onclick="runTests()">Run evaluation</button>

<pre id="testresult"></pre>

</body>
</html>
"""


# ---------------------------------------------------------------------
# Proxy endpoints
# ---------------------------------------------------------------------

@app.get("/proxy/search")
async def proxy_search(query: str):
    res = await call_service("/v1/search", {"query": query})
    return res


@app.get("/proxy/similar/{course_id}")
async def proxy_similar(course_id: str):
    res = await call_service(f"/v1/courses/{course_id}/similar", {})
    return res


@app.get("/proxy/objectives")
async def proxy_objectives(query: str):
    res = await call_service("/v1/objectives/search", {"query": query})
    return res


@app.get("/evaluate")
async def evaluate():
    return await evaluate_dataset()
