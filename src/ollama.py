from __future__ import annotations

import json
import urllib.error
import urllib.request


def ask_ollama(model: str, prompt: str, host: str, timeout: int = 120) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 1},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot reach Ollama at {url}: {error}") from error

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON from Ollama: {body[:300]}") from error

    model_response = parsed.get("response")
    if not isinstance(model_response, str):
        raise RuntimeError("Ollama response did not include text in 'response'.")
    return model_response.strip()
