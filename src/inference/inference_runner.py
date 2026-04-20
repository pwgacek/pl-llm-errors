from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .llm_client import LLMClient

from src.settings import settings


Prompts = dict[str, dict[str, list[dict]]]


def _sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", value)
    sanitized = re.sub(r"\s+", "_", sanitized).strip(" ._")
    return sanitized or "model"


def _build_report_path(model: str, timestamp: str) -> Path:
    safe_model_name = _sanitize_filename_component(str(model))
    return Path("results") / f"{safe_model_name}_{timestamp}.json"


def _load_jsonl_records(prompt_file: Path) -> list[dict]:
    records: list[dict] = []
    for raw_line in prompt_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def step_load_prompts(prompts_dir: Path) -> Prompts:
    print(f"\n=== Step 1: Loading prompts from {prompts_dir} ===")
    prompts: Prompts = {}
    if not prompts_dir.is_dir():
        raise ValueError(f"  Prompts directory '{prompts_dir}' not found.")

    generator_dirs = sorted(path for path in prompts_dir.iterdir() if path.is_dir())
    for gen_dir in generator_dirs:
        gen_name = gen_dir.name
        datasets: dict[str, list[dict]] = {}

        for prompt_file in sorted(gen_dir.glob("*.jsonl")):
            dataset_name = prompt_file.stem
            records = _load_jsonl_records(prompt_file)
            datasets[dataset_name] = records
            print(f"  [{gen_name}/{dataset_name}] Loaded {len(records)} prompts.")

        prompts[gen_name] = datasets

    return prompts


def _save_partial(report_path: Path, report: dict) -> None:
    tmp = report_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(report_path)


def _ask_records(
    records: list[dict],
    model_client: LLMClient,
    workers: int,
) -> list[dict]:
    total = len(records)

    def ask_one(args: tuple[int, dict]) -> dict:
        i, record = args
        prompt_id = str(record.get("id", f"row-{i:03d}"))
        prompt = record["prompt"]
        judgement_context = record.get("judgement_context", {})
        t0 = time.perf_counter()

        try:
            raw = model_client.ask(prompt)
        except Exception as e:
            print(f"    [{i}/{total}] Model error: {e}")
            return {
                "index": i,
                "prompt_id": prompt_id,
                "raw_answer": "",
                "elapsed": 0.0,
                "error": True,
                "judgement_context": judgement_context,
            }

        elapsed = time.perf_counter() - t0
        print(f"    [{i}/{total}] ok  ({elapsed:.2f}s)")
        return {
            "index": i,
            "prompt_id": prompt_id,
            "raw_answer": raw,
            "elapsed": round(elapsed, 3),
            "error": False,
            "judgement_context": judgement_context,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(ask_one, (i, r)): i for i, r in enumerate(records, start=1)}
        raw_records = [future.result() for future in as_completed(futures)]

    raw_records.sort(key=lambda record: record["index"])
    return [{k: v for k, v in record.items() if k != "index"} for record in raw_records]


def step_run(
    prompts: Prompts,
    model_client: LLMClient,
    workers: int,
    report_path: Path,
    report_skeleton: dict,
    existing_answers: dict[str, dict] | None = None,
) -> dict[str, dict]:
    print("\n=== Step 2: Asking LLM ===")
    answers: dict[str, dict] = existing_answers if existing_answers is not None else {}

    for gen_name, datasets in prompts.items():
        if gen_name not in answers:
            answers[gen_name] = {}

        for dataset_name, records in datasets.items():
            if dataset_name in answers[gen_name]:
                print(f"\n  [{gen_name}/{dataset_name}] Already completed, skipping (resume).")
                continue

            total = len(records)
            print(f"\n  [{gen_name}/{dataset_name}] Asking {total} questions ...")
            dataset_start = time.perf_counter()

            output_records = _ask_records(records, model_client, workers)
            error_count = sum(1 for r in output_records if r["error"])
            dataset_elapsed = time.perf_counter() - dataset_start
            print(
                f"  [{gen_name}/{dataset_name}] errors={error_count}  time={dataset_elapsed:.1f}s"
            )

            answers[gen_name][dataset_name] = {
                "num_sampled": total,
                "error_count": error_count,
                "elapsed_sec": round(dataset_elapsed, 2),
                "questions": output_records,
            }

            partial = {**report_skeleton, "datasets": answers, "status": "partial"}
            _save_partial(report_path, partial)
            print(f"  Progress saved to {report_path}")

    return answers


def main() -> None:
    model = settings.inference.model
    base_url = settings.inference.base_url
    api_key = settings.inference.api_key
    seed = int(settings.common.seed)
    temperature = int(settings.inference.temperature)
    prompts_dir = Path(settings.common.prompt_dir)
    workers = int(settings.inference.workers)
    resume = settings.inference.resume
    timeout = int(settings.inference.timeout)

    llm_client = LLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        seed=seed,
        temperature=temperature,
        timeout=timeout,
    )

    pipeline_start = time.perf_counter()

    prompts = step_load_prompts(prompts_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    report_path = _build_report_path(str(model), timestamp)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_skeleton = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "workers": workers,
        "prompts_dir": str(prompts_dir),
    }

    existing_answers: dict[str, dict] | None = None
    if resume:
        resume_path = Path(str(resume))
        if resume_path.exists():
            prev = json.loads(resume_path.read_text(encoding="utf-8"))
            existing_answers = prev.get("datasets", {})
            report_path = resume_path
            done = sum(len(ds) for ds in existing_answers.values())
            print(f"  Resuming from {resume_path} ({done} generator+dataset pairs already done.)")  
        else:
            print(f"  Resume file {resume_path} not found, starting fresh.")

    answers = step_run(
        prompts, llm_client, workers, report_path, report_skeleton, existing_answers,
    )

    print("\n=== Summary ===")
    for gen_name, datasets in answers.items():
        total_samples = sum(d["num_sampled"] for d in datasets.values())
        total_errors = sum(d["error_count"] for d in datasets.values())
        print(f"  [{gen_name}] total={total_samples}  errors={total_errors}")

    report = {
        **report_skeleton,
        "elapsed_sec": round(time.perf_counter() - pipeline_start, 2),
        "datasets": answers,
        "status": "complete",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
