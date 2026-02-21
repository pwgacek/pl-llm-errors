from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from load_matura_math import load_matura_math_questions


def build_prompt(question_obj: dict[str, Any]) -> str:
    question = question_obj["question"]
    answers = question_obj["answers"]

    letters = ["A", "B", "C", "D"]
    choices = "\n".join(f"{letters[i]}. {answer}" for i, answer in enumerate(answers))

    return (
        "Przemyśl pytanie krok po kroku, a następnie wybierz poprawną odpowiedź spośród 4 możliwych.\n"
        "Odpowiedz w formacie: {\"odpowiedź\": \"LITERA\"}\n"
        f"<PYTANIE>:{question}</PYTANIE>\n"
        f"<ODPOWIEDZI>:{choices}</ODPOWIEDZI>\n"
    )


def call_ollama(model: str, prompt: str, host: str, timeout: int = 120) -> str:
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


def extract_answer_index(text: str) -> int | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    answer = payload.get("odpowiedź")
    if not isinstance(answer, str):
        return None

    normalized = answer.strip().upper()
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(normalized)

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask random Matematyka questions to a local Ollama model and report aggregate stats."
    )
    parser.add_argument(
        "--input",
        default="datasets/llmzszl.jsonl",
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--model",
        default="SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="How many random questions to ask (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.input)
    questions = load_matura_math_questions(dataset_path)
    if not questions:
        print("No Matematyka questions found.")
        sys.exit(1)

    num_questions = max(1, args.num_questions)
    sample_size = min(num_questions, len(questions))

    if sample_size < num_questions:
        print(
            f"Requested {num_questions} questions, but only {len(questions)} are available. "
            f"Using {sample_size}."
        )

    rng = random.Random(args.seed)
    sampled_questions = rng.sample(questions, sample_size)

    correct = 0
    incorrect = 0
    parsing_error = 0

    print(
        f"Sending {sample_size} random questions to Ollama "
        f"(seed={args.seed}, model={args.model})..."
    )

    total_start = time.perf_counter()

    for question_no, question_obj in enumerate(sampled_questions, start=1):
        prompt = build_prompt(question_obj)
        question_start = time.perf_counter()
        model_raw_answer = call_ollama(args.model, prompt, args.host)
        question_elapsed = time.perf_counter() - question_start

        predicted_index = extract_answer_index(model_raw_answer)
        correct_index = question_obj["correct_answer_index"]

        print(f"\n[{question_no}/{sample_size}] {question_obj['question']}")
        print(f"Model raw response: {model_raw_answer}")
        print(f"Time: {question_elapsed:.2f}s")

        if predicted_index is None:
            parsing_error += 1
            print("Result: PARSING_ERROR ⚠️")
            continue

        if predicted_index == correct_index:
            correct += 1
            print("Result: CORRECT ✅")
        else:
            incorrect += 1
            print("Result: INCORRECT ❌")

    print("\n=== Summary ===")
    print(f"correct: {correct}")
    print(f"incorrect: {incorrect}")
    print(f"parsing_error: {parsing_error}")
    print(f"total_time_sec: {time.perf_counter() - total_start:.2f}")


if __name__ == "__main__":
    main()
