#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request

input_file = Path("src/scripts/translate/gsm-hard.jsonl")
output_file = Path("src/scripts/translate/gsm-hard-pl.json")

ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
ollama_model = os.getenv("OLLAMA_MODEL", "translategemma:12b")

# Set to an int for testing, or None for unlimited.
max_translations = None


def _load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def translate_question(question: str) -> dict[str, object]:
    prompt = (
        "You are an expert translator specializing in mathematical word problems. "
        "Translate the following math problem from English to Polish. "
        "Follow these strict rules:\n"
        "1. Keep all numbers, constants, and variables exactly unchanged.\n"
        "2. Maintain the exact mathematical logic, relations, and operations.\n"
        "3. Ensure perfect Polish grammar, especially correct verb and noun inflections when they follow large numbers or fractions.\n"
        "4. Use natural Polish mathematical phrasing.\n"
        "5. Provide ONLY the final Polish translation text. Do not include any introductions, explanations, or commentary.\n\n"
        f"Problem to translate:\n{question}"
    )

    try:
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        req = urllib.request.Request(
            ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as http_err:
            body = http_err.read().decode("utf-8", errors="replace")
            return {"success": False, "error": f"HTTP {http_err.code}: {body[:500]}"}

        data = json.loads(raw)
        translated = data.get("response")
        if not isinstance(translated, str) or not translated.strip():
            return {"success": False, "error": "Empty response from Ollama"}

        return {"success": True, "translated": translated.strip()}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def main() -> None:
    results: list[dict] = _load_existing_results(output_file)
    if results:
        print(f"Loaded {len(results)} existing translations")
    else:
        print("Starting fresh translation")

    translated_indices = {item.get("index") for item in results if item.get("index") is not None}
    translated_count = 0

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        return

    with input_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_translations is not None and translated_count >= max_translations:
                print(f"\nReached limit of {max_translations} translations")
                break

            if idx in translated_indices:
                print(f"[{idx}] Already translated, skipping")
                continue

            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"[{idx}] Invalid JSON, skipping")
                continue

            question = record.get("question", "")
            answer = record.get("answer", "")
            if not question:
                print(f"[{idx}] Empty question, skipping")
                continue

            translated_count += 1
            print(f"[{idx}] Translating... ", end="", flush=True)
            trans_result = translate_question(question=question)

            success = bool(trans_result.get("success"))
            translated_text = trans_result.get("translated") if success else None
            error = trans_result.get("error")

            results.append(
                {
                    "index": idx,
                    "original_question": question,
                    "translated_question": translated_text,
                    "answer": answer,
                    "success": success,
                    "error": error,
                }
            )
            _save_results(output_file, results)

            status = "✓" if success else "✗"
            print(status)

    print(f"\nCompleted! Saved {len(results)} translations to {output_file}")
    successful = sum(1 for r in results if r.get("success"))
    print(f"Successful: {successful}/{len(results)}")


if __name__ == "__main__":
    main()
