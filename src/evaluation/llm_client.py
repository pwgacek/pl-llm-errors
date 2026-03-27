from __future__ import annotations

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        seed: int,
        temperature: int,
    ) -> None:
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)

    def ask(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Jesteś pomocnym asystentem, który odpowiada na "
                        "pytania w formacie JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            seed=self.seed,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("No content in model response.")
        return content.strip()
