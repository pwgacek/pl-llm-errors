from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .llm_client import LLMClient

from src.settings import settings


Prompts = dict[str, dict[str, list[dict]]]
POLISH_TZ = ZoneInfo("Europe/Warsaw")


def _sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", value)
    sanitized = re.sub(r"\s+", "_", sanitized).strip(" ._")
    return sanitized or "model"


def _build_report_path(model: str, timestamp: str) -> Path:
    safe_model_name = _sanitize_filename_component(str(model))
    answers_dir = Path(str(settings.common.answers_dir))
    return answers_dir / safe_model_name / f"{timestamp}.json"


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


def step_run(
    prompts: Prompts,
    model_client: LLMClient,
    report_path: Path,
    report_skeleton: dict,
) -> tuple[dict[str, dict], bool]:
    print("\n=== Step 2: Asking LLM ===")
    answers: dict[str, dict] = {}
    processed_any = False

    for gen_name, datasets in prompts.items():
        if gen_name not in answers:
            answers[gen_name] = {}

        for dataset_name, records in datasets.items():
            total = len(records)
            output_records: list[dict] = []
            dataset_start = time.perf_counter()
            print(f"\n  [{gen_name}/{dataset_name}] Asking {total} questions ...")

            for i, record in enumerate(records, start=1):
                prompt_id = str(record.get("id", f"row-{i:03d}"))
                processed_any = True

                prompt = record["prompt"]
                judgement_context = record.get("judgement_context", {})
                t0 = time.perf_counter()

                try:
                    raw = model_client.ask(prompt)
                    elapsed = time.perf_counter() - t0
                    print(f"    [{i}/{total}] ok  ({elapsed:.2f}s)")
                    current = {
                        "prompt_id": prompt_id,
                        "raw_answer": raw,
                        "elapsed": round(elapsed, 3),
                        "error": False,
                        "judgement_context": judgement_context,
                    }
                except Exception as e:
                    print(f"    [{i}/{total}] Model error: {e}")
                    current = {
                        "prompt_id": prompt_id,
                        "raw_answer": "",
                        "elapsed": 0.0,
                        "error": True,
                        "judgement_context": judgement_context,
                    }

                output_records.append(current)

                error_count = sum(1 for r in output_records if r["error"])
                answers[gen_name][dataset_name] = {
                    "num_sampled": total,
                    "error_count": error_count,
                    "elapsed_sec": round(time.perf_counter() - dataset_start, 2),
                    "questions": output_records,
                }

                partial = {**report_skeleton, "datasets": answers, "status": "partial"}
                _save_partial(report_path, partial)

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

    return answers, processed_any


def main() -> None:
    model = settings.inference.model
    base_url = settings.inference.base_url
    api_key = settings.inference.api_key
    seed = int(settings.common.seed)
    temperature = int(settings.inference.temperature)
    prompts_dir = Path(settings.common.prompt_dir)
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

    timestamp = datetime.now(POLISH_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    report_path = _build_report_path(str(model), timestamp)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_skeleton = {
        "timestamp": datetime.now(POLISH_TZ).isoformat(),
        "model": model,
        "base_url": base_url,
        "prompts_dir": str(prompts_dir),
    }

    answers, processed_any = step_run(
        prompts, llm_client, report_path, report_skeleton,
    )

    print("\n=== Summary ===")
    for gen_name, datasets in answers.items():
        total_samples = sum(d["num_sampled"] for d in datasets.values())
        total_errors = sum(d["error_count"] for d in datasets.values())
        print(f"  [{gen_name}] total={total_samples}  errors={total_errors}")

    # Collect and write failed records organized by generator and dataset
    prompts_to_rerun_base = Path(str(settings.common.prompt_dir) + "_to_rerun")
    model_subdir = prompts_to_rerun_base / _sanitize_filename_component(str(model)) / timestamp
    
    failed_records_by_dataset: dict[tuple[str, str], list[dict]] = {}
    
    for gen_name, datasets in answers.items():
        for dataset_name, dataset_info in datasets.items():
            questions = dataset_info.get("questions", [])
            for i, question in enumerate(questions):
                if question.get("error"):
                    if gen_name in prompts and dataset_name in prompts[gen_name]:
                        prompt_record = prompts[gen_name][dataset_name][i]
                        key = (gen_name, dataset_name)
                        if key not in failed_records_by_dataset:
                            failed_records_by_dataset[key] = []
                        failed_records_by_dataset[key].append(prompt_record)

    # Write failed records to organized files
    total_failed = 0
    if failed_records_by_dataset:
        for (gen_name, dataset_name), records in failed_records_by_dataset.items():
            gen_dir = model_subdir / gen_name
            gen_dir.mkdir(parents=True, exist_ok=True)
            
            rerun_path = gen_dir / f"{dataset_name}.jsonl"
            
            with open(rerun_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            total_failed += len(records)
            print(f"\nFailed records written to: {rerun_path} ({len(records)} records)")
        
        print(f"\nTotal failed records: {total_failed}")


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
