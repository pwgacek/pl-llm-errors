from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from download import download_file
from loaders import BelebeleLoader, CDSLoader, LDEKLoader, LLMZSZLLoader, PolQALoader
from ollama import ask_ollama
from questions import VerificationResult

DATASETS = [
    {
        "name": "llmzszl",
        "url": "https://huggingface.co/datasets/amu-cai/llmzszl-dataset/resolve/main/llmzszl-test.jsonl",
        "output": Path("datasets/llmzszl.jsonl"),
        "loader": LLMZSZLLoader,
    },
    {
        "name": "belebele-pol",
        "url": "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/pol_Latn.jsonl",
        "output": Path("datasets/belebele-pol.jsonl"),
        "loader": BelebeleLoader,
    },
    {
        "name": "polqa",
        "url": "https://huggingface.co/datasets/ipipan/polqa/resolve/main/data/test.csv",
        "output": Path("datasets/polqa.csv"),
        "loader": PolQALoader,
    },
    {
        "name": "cds",
        "url": "https://git.nlp.ipipan.waw.pl/Scwad/SCWAD-CDSCorpus/repository/archive.zip",
        "output": Path("datasets/CDS_test.csv"),
        "loader": CDSLoader,
    },
    {
        "name": "ldek",
        "url": "https://huggingface.co/datasets/amu-cai/medical-exams-LDEK-PL-2008-2024/resolve/main/medical-exams-LDEK-PL-2008-2024.json",
        "output": Path("datasets/medical-exams-LDEK-PL-2008-2024.json"),
        "loader": LDEKLoader,
    },
]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_download() -> None:
    print("\n=== Step 1: Downloading datasets ===")
    downloaded = skipped = failed = 0
    for dataset in DATASETS:
        name, url, output = dataset["name"], dataset["url"], dataset["output"]
        if output.exists():
            print(f"  [{name}] Already exists, skipping.")
            skipped += 1
            continue
        print(f"  [{name}] Downloading {url} ...")
        try:
            download_file(url, output)
            print(f"  [{name}] Done.")
            downloaded += 1
        except urllib.error.URLError as e:
            print(f"  [{name}] FAILED: {e}")
            failed += 1

    print(f"  downloaded={downloaded}  skipped={skipped}  failed={failed}")
    if failed > 0:
        print("Some downloads failed. Proceeding with available datasets.")


def step_load() -> dict[str, list]:
    print("\n=== Step 2: Loading datasets ===")
    loaded: dict[str, list] = {}
    for dataset in DATASETS:
        name = dataset["name"]
        loader_cls = dataset["loader"]
        if not dataset["output"].exists():
            print(f"  [{name}] File missing, skipping.")
            continue
        try:
            questions = loader_cls().load()
            loaded[name] = questions
            print(f"  [{name}] Loaded {len(questions)} questions.")
        except Exception as e:
            print(f"  [{name}] Load error: {e}")
    return loaded


def step_evaluate(
    loaded: dict[str, list],
    model: str,
    host: str,
    num_questions: int,
    seed: int,
) -> dict[str, dict]:
    print("\n=== Step 3: Evaluating with Ollama ===")
    rng = random.Random(seed)
    results: dict[str, dict] = {}

    for name, questions in loaded.items():
        sample_size = min(num_questions, len(questions))
        sampled = rng.sample(questions, sample_size)
        print(f"\n  [{name}] Asking {sample_size}/{len(questions)} questions ...")

        correct = incorrect = error = 0
        records = []
        dataset_start = time.perf_counter()

        for i, q in enumerate(sampled, start=1):
            prompt = q.build_prompt()
            t0 = time.perf_counter()
            try:
                raw = ask_ollama(model, prompt, host)
            except RuntimeError as e:
                print(f"    [{i}/{sample_size}] Ollama error: {e}")
                error += 1
                records.append({"question": q.build_prompt(), "result": "ERROR", "elapsed": 0.0})
                continue
            elapsed = time.perf_counter() - t0

            verdict = q.verify_answer(raw)
            label = verdict.name  # "CORRECT" / "INCORRECT" / "ERROR"
            if verdict == VerificationResult.CORRECT:
                correct += 1
            elif verdict == VerificationResult.INCORRECT:
                incorrect += 1
            else:
                error += 1

            symbol = {"CORRECT": "✅", "INCORRECT": "❌", "ERROR": "⚠️"}.get(label, "")
            print(f"    [{i}/{sample_size}] {label} {symbol}  ({elapsed:.2f}s)")
            records.append({"question": q.build_prompt(), "raw_answer": raw, "result": label, "elapsed": round(elapsed, 3)})

        dataset_elapsed = time.perf_counter() - dataset_start
        total = correct + incorrect + error
        accuracy = correct / total if total > 0 else 0.0
        print(f"  [{name}] correct={correct}  incorrect={incorrect}  error={error}  accuracy={accuracy:.2%}  time={dataset_elapsed:.1f}s")

        results[name] = {
            "num_sampled": sample_size,
            "correct": correct,
            "incorrect": incorrect,
            "error": error,
            "accuracy": accuracy,
            "elapsed_sec": round(dataset_elapsed, 2),
            "questions": records,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: download → load → evaluate → report.")
    parser.add_argument("--model", default="SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0", help="Ollama model name.")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--num-questions", type=int, default=20, help="Random questions per dataset (default: 20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--report", default="results/report.json", help="Output report file path (default: report.json).")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download step.")
    args = parser.parse_args()

    pipeline_start = time.perf_counter()

    if not args.skip_download:
        step_download()

    loaded = step_load()
    if not loaded:
        print("No datasets loaded. Exiting.")
        sys.exit(1)

    results = step_evaluate(loaded, args.model, args.host, args.num_questions, args.seed)

    # Overall summary
    total_correct = sum(r["correct"] for r in results.values())
    total_incorrect = sum(r["incorrect"] for r in results.values())
    total_error = sum(r["error"] for r in results.values())
    total_all = total_correct + total_incorrect + total_error
    overall_accuracy = total_correct / total_all if total_all > 0 else 0.0

    print("\n=== Overall Summary ===")
    print(f"  correct={total_correct}  incorrect={total_incorrect}  error={total_error}  accuracy={overall_accuracy:.2%}")

    # Write report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "num_questions_per_dataset": args.num_questions,
        "overall": {
            "correct": total_correct,
            "incorrect": total_incorrect,
            "error": total_error,
            "accuracy": overall_accuracy,
            "elapsed_sec": round(time.perf_counter() - pipeline_start, 2),
        },
        "datasets": results,
    }

    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
