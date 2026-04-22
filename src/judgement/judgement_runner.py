from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.inference.llm_client import LLMClient
from src.settings import settings
from .prompts import build_bbh_judge_prompt, build_llmzszl_judge_prompt, build_polqa_judge_prompt


def _build_judge_prompt(dataset_name: str, model_answer: str, judgement_context: dict) -> str | None:
    if dataset_name == "llmzszl":
        return build_llmzszl_judge_prompt(
            question=judgement_context["question"],
            correct_answer=judgement_context["correct_answer"],
            model_answer=model_answer,
        )
    if dataset_name == "polqa":
        return build_polqa_judge_prompt(
            question=judgement_context["question"],
            context=judgement_context["context"],
            correct_answer=judgement_context["correct_answer"],
            model_answer=model_answer,
        )
    if dataset_name == "bbh":
        return build_bbh_judge_prompt(
            question=judgement_context["question"],
            correct_answer=judgement_context["correct_answer"],
            model_answer=model_answer,
        )
    return None


def _judge_records(
    records: list[dict],
    dataset_name: str,
    judge_client: LLMClient,
    workers: int,
) -> list[dict]:
    total = len(records)

    def judge_one(args: tuple[int, dict]) -> dict:
        i, record = args
        model_answer = record.get("raw_answer", "")
        judgement_context = record.get("judgement_context", {})

        judge_prompt = _build_judge_prompt(dataset_name, model_answer, judgement_context)
        if judge_prompt is None:
            return {**record, "index": i, "judge_scores": None, "judge_error": True}

        t0 = time.perf_counter()
        try:
            result = judge_client.ask(judge_prompt, json_mode=True)
        except Exception as e:
            print(f"    [{i}/{total}] Judge error: {e}")
            return {**record, "index": i, "judge_scores": None, "judge_error": True}

        elapsed = time.perf_counter() - t0
        print(f"    [{i}/{total}] {result}  ({elapsed:.2f}s)")
        return {**record, "index": i, "judge_scores": result, "judge_error": False}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge_one, (i, r)): i for i, r in enumerate(records, start=1)}
        results = [future.result() for future in as_completed(futures)]

    results.sort(key=lambda r: r["index"])
    return [{k: v for k, v in r.items() if k != "index"} for r in results]


def _save(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(path)


def run_judgement(
    answers_path: Path,
    judge_client: LLMClient,
    workers: int,
    output_path: Path,
    existing_judgements: dict[str, dict] | None = None,
) -> None:
    print(f"\n=== Judgement: loading {answers_path} ===")
    report = json.loads(answers_path.read_text(encoding="utf-8"))
    datasets: dict[str, dict] = report.get("datasets", {})

    judgements: dict[str, dict] = existing_judgements if existing_judgements is not None else {}

    for gen_name, gen_datasets in datasets.items():
        if gen_name not in judgements:
            judgements[gen_name] = {}

        for dataset_name, dataset_data in gen_datasets.items():
            if dataset_name in judgements[gen_name]:
                print(f"\n  [{gen_name}/{dataset_name}] Already completed, skipping (resume).")
                continue

            print(f"\n  [{gen_name}/{dataset_name}] Judging ...")
            records: list[dict] = dataset_data.get("questions", [])
            judged = _judge_records(records, dataset_name, judge_client, workers)
            error_count = sum(1 for r in judged if r["judge_error"])
            judgements[gen_name][dataset_name] = {
                **{k: v for k, v in dataset_data.items() if k != "questions"},
                "judge_error_count": error_count,
                "questions": judged,
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            _save(
                output_path,
                {
                    "source": str(answers_path),
                    "judge_model": judge_client.model,
                    "datasets": judgements,
                    "status": "partial",
                },
            )
            print(f"  Progress saved to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save(
        output_path,
        {
            "source": str(answers_path),
            "judge_model": judge_client.model,
            "datasets": judgements,
            "status": "complete",
        },
    )

    print(f"\n=== Judgement complete: {output_path} ===")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.judgement.judgement_runner <answers_file.json>")
        sys.exit(1)

    answers_path = Path(sys.argv[1])
    if not answers_path.exists():
        print(f"File not found: {answers_path}")
        sys.exit(1)

    model = settings.judgement.model
    base_url = settings.judgement.base_url
    api_key = settings.judgement.api_key
    seed = int(settings.common.seed)
    temperature = int(settings.judgement.temperature)
    workers = int(settings.judgement.workers)
    resume = settings.judgement.resume
    timeout = int(settings.judgement.timeout)
    judgements_dir = Path(str(settings.common.judgements_dir))

    judge_client = LLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        seed=seed,
        temperature=temperature,
        timeout=timeout,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    output_path = judgements_dir / f"{answers_path.stem}_judged_{timestamp}.json"
    existing_judgements: dict[str, dict] | None = None

    if resume:
        resume_path = Path(str(resume))
        if resume_path.exists():
            prev = json.loads(resume_path.read_text(encoding="utf-8"))
            existing_judgements = prev.get("datasets", {})
            output_path = resume_path
            done = sum(len(ds) for ds in existing_judgements.values())
            print(f"  Resuming from {resume_path} ({done} generator+dataset pairs already done.)")
        else:
            print(f"  Resume file {resume_path} not found, starting fresh.")

    run_judgement(answers_path, judge_client, workers, output_path, existing_judgements)


if __name__ == "__main__":
    main()