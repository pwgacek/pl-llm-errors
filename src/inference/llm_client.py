from __future__ import annotations

import json

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        seed: int,
        temperature: int,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def ask(self, prompt: str, json_mode: bool = False) -> str | dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            seed=self.seed,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("No content in model response.")
        return json.loads(content) if json_mode else content.strip()
