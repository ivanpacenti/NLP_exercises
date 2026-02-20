import os
import requests


API_URL = "https://chat.campusai.compute.dtu.dk/api/chat/completions"


def send_message(prompt, model="Gemma3", temperature=0.0, timeout=30):
    api_key = os.getenv("CAMPUS_AI_API_KEY") or os.getenv("CAMPUSAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing CAMPUS_AI_API_KEY (or CAMPUSAI_API_KEY) in environment")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }

    response = requests.post(API_URL, json=payload, headers=headers, timeout=timeout)
    try:
        data = response.json()
    except ValueError:
        data = {"_raw": response.text, "_status": response.status_code}

    if not response.ok:
        raise RuntimeError(f"CampusAI error {response.status_code}: {data}")
    return data
