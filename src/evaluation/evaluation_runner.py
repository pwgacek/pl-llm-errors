from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from model import ask_model

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(raw: str, expected: dict) -> str:
    """Return 'CORRECT', 'INCORRECT', or 'ERROR' by comparing the model's raw
    response against the *expected* dict saved by PromptBuilder."""
    kind = expected.get("type")

    if kind == "multiple_choice_index":
        predicted = _extract_letter(raw)
        if predicted is None:
            return "ERROR"
        correct_letter = "ABCD"[expected["correct_index"]]
        return "CORRECT" if predicted == correct_letter else "INCORRECT"

    if kind == "multiple_choice_letter":
        predicted = _extract_letter(raw)
        if predicted is None:
            return "ERROR"
        return "CORRECT" if predicted == expected["correct_letter"].upper() else "INCORRECT"

    if kind == "open_contained":
        normalized = raw.strip().lower()
        if not normalized:
            return "ERROR"
        for accepted in expected["accepted_answers"]:
            if accepted.strip().lower() in normalized:
                return "CORRECT"
        return "INCORRECT"

    if kind == "entailment":
        answer = _extract_json_field(raw, ("odpowiedź", "odpowiedz", "answer"))
        if answer is None:
            answer = raw.strip().upper()
        else:
            answer = answer.strip().upper()
        if answer not in {"NEUTRAL", "CONTRADICTION", "ENTAILMENT"}:
            return "ERROR"
        return "CORRECT" if answer == expected["judgment"].upper() else "INCORRECT"

    return "ERROR"


def _extract_letter(text: str) -> str | None:
    """Parse a JSON response and return the answer letter (A-E), or None."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    raw_answer = _extract_json_field_from_dict(payload, ("odpowiedź", "odpowiedz", "answer"))
    if raw_answer is None:
        return None
    if isinstance(raw_answer, int) and 0 <= raw_answer <= 4:
        return "ABCDE"[raw_answer]
    if not isinstance(raw_answer, str):
        return None
    normalized = raw_answer.strip().upper()
    return normalized if normalized in "ABCDE" and len(normalized) == 1 else None


def _extract_json_field(text: str, keys: tuple[str, ...]) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _extract_json_field_from_dict(payload, keys)


def _extract_json_field_from_dict(payload: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

# Prompts structure loaded from disk:
#   dict[generator_name, dict[dataset_name, list[{"prompt": str, "expected": dict}]]]
Prompts = dict[str, dict[str, list[dict]]]


def step_load_prompts(prompts_dir: Path) -> Prompts:
    print(f"\n=== Step 1: Loading prompts from {prompts_dir} ===")
    prompts: Prompts = {}
    if not prompts_dir.is_dir():
        print(f"  Prompts directory '{prompts_dir}' not found.")
        return prompts

    for gen_dir in sorted(prompts_dir.iterdir()):
        if not gen_dir.is_dir():
            continue
        gen_name = gen_dir.name
        prompts[gen_name] = {}
        for prompt_file in sorted(gen_dir.glob("*.jsonl")):
            dataset_name = prompt_file.stem
            records = []
            for line in prompt_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            prompts[gen_name][dataset_name] = records
            print(f"  [{gen_name}/{dataset_name}] Loaded {len(records)} prompts.")

    return prompts


def _save_partial(report_path: Path, report: dict) -> None:
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(report_path)


def step_evaluate(
    prompts: Prompts,
    model: str,
    base_url: str,
    api_key: str,
    workers: int = 1,
    report_path: Path | None = None,
    report_skeleton: dict | None = None,
    existing_results: dict[str, dict] | None = None,
) -> dict[str, dict]:
    print("\n=== Step 2: Evaluating ===")
    results: dict[str, dict] = existing_results if existing_results is not None else {}

    for gen_name, datasets in prompts.items():
        if gen_name not in results:
            results[gen_name] = {}

        for dataset_name, records in datasets.items():
            # Skip pairs already completed when resuming
            if dataset_name in results[gen_name]:
                print(f"\n  [{gen_name}/{dataset_name}] Already completed, skipping (resume).")
                continue

            total = len(records)
            print(f"\n  [{gen_name}/{dataset_name}] Asking {total} questions ...")
            dataset_start = time.perf_counter()

            def ask_one(args: tuple[int, dict]) -> dict:
                i, record = args
                prompt = record["prompt"]
                expected = record["expected"]
                t0 = time.perf_counter()
                try:
                    raw = ask_model(model, prompt, base_url, api_key)
                except Exception as e:
                    print(f"    [{i}/{total}] Model error: {e}")
                    return {"index": i, "prompt": prompt, "result": "ERROR", "elapsed": 0.0}
                elapsed = time.perf_counter() - t0
                label = _verify(raw, expected)
                symbol = {"CORRECT": "✅", "INCORRECT": "❌", "ERROR": "⚠️"}.get(label, "")
                print(f"    [{i}/{total}] {label} {symbol}  ({elapsed:.2f}s)")
                return {"index": i, "prompt": prompt, "raw_answer": raw, "result": label, "elapsed": round(elapsed, 3)}

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(ask_one, (i, r)): i for i, r in enumerate(records, start=1)}
                raw_records = [f.result() for f in as_completed(futures)]

            raw_records.sort(key=lambda r: r["index"])
            output_records = [{k: v for k, v in r.items() if k != "index"} for r in raw_records]

            correct = sum(1 for r in output_records if r["result"] == "CORRECT")
            incorrect = sum(1 for r in output_records if r["result"] == "INCORRECT")
            error = sum(1 for r in output_records if r["result"] == "ERROR")
            dataset_elapsed = time.perf_counter() - dataset_start
            accuracy = correct / total if total > 0 else 0.0
            print(f"  [{gen_name}/{dataset_name}] correct={correct}  incorrect={incorrect}  error={error}  accuracy={accuracy:.2%}  time={dataset_elapsed:.1f}s")

            results[gen_name][dataset_name] = {
                "num_sampled": total,
                "correct": correct,
                "incorrect": incorrect,
                "error": error,
                "accuracy": accuracy,
                "elapsed_sec": round(dataset_elapsed, 2),
                "questions": output_records,
            }

            if report_path is not None and report_skeleton is not None:
                partial = {**report_skeleton, "datasets": results, "status": "partial"}
                _save_partial(report_path, partial)
                print(f"  💾 Progress saved to {report_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation pipeline: read prompts → ask model → report.")
    parser.add_argument("--model", default="SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0", help="Model name.")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default="ollama", help="API key.")
    parser.add_argument("--prompts-dir", default="prompts", help="Directory with pre-built prompts (default: prompts/).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for model requests (default: 1).")
    parser.add_argument("--report", default="results/report.json", help="Output report file path (default: results/report.json).")
    parser.add_argument("--resume", type=str, default=None, metavar="REPORT", help="Resume from a partial report JSON file.")
    args = parser.parse_args()

    pipeline_start = time.perf_counter()

    prompts = step_load_prompts(Path(args.prompts_dir))
    if not prompts:
        print("No prompts loaded. Run prompt_builder.py first.")
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    base_path = Path(args.report)
    report_path = base_path.parent / f"{base_path.stem}_{timestamp}{base_path.suffix}"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_skeleton = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "workers": args.workers,
        "prompts_dir": args.prompts_dir,
    }

    existing_results: dict[str, dict] | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            prev = json.loads(resume_path.read_text(encoding="utf-8"))
            existing_results = prev.get("datasets", {})
            report_path = resume_path
            done = sum(len(ds) for ds in existing_results.values())
            print(f"  Resuming from {resume_path} ({done} generator+dataset pairs already done).")
        else:
            print(f"  Resume file {resume_path} not found, starting fresh.")

    results = step_evaluate(
        prompts, args.model, args.base_url, args.api_key, args.workers,
        report_path=report_path, report_skeleton=report_skeleton, existing_results=existing_results,
    )

    # Overall summary per generator
    print("\n=== Overall Summary ===")
    for gen_name, datasets in results.items():
        total_correct = sum(d["correct"] for d in datasets.values())
        total_incorrect = sum(d["incorrect"] for d in datasets.values())
        total_error = sum(d["error"] for d in datasets.values())
        total_all = total_correct + total_incorrect + total_error
        overall_accuracy = total_correct / total_all if total_all > 0 else 0.0
        print(f"  [{gen_name}] correct={total_correct}  incorrect={total_incorrect}  error={total_error}  accuracy={overall_accuracy:.2%}")

    report = {
        **report_skeleton,
        "elapsed_sec": round(time.perf_counter() - pipeline_start, 2),
        "datasets": results,
        "status": "complete",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
