from __future__ import annotations

from openai import OpenAI


def ask_model(model: str, prompt: str, base_url: str, api_key: str = "ollama", timeout: int = 120) -> str:
    """Call any OpenAI-compatible endpoint (Ollama or vLLM).

    Ollama:  base_url="http://localhost:11434/v1", api_key="ollama"
    vLLM:    base_url="http://<server>:8000/v1",   api_key=<whatever is set>
    """
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("No content in model response.")
    return content.strip()
