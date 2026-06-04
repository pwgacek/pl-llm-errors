from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.inference.llm_client import LLMClient
from src.settings import settings
from .prompts import (
    build_bbh_judge_prompt,
    build_matura_judge_prompt,
)


POLISH_TZ = ZoneInfo("Europe/Warsaw")


def _build_judge_prompt(dataset_name: str, model_answer: str, judgement_context: dict) -> str | None:
    if dataset_name == "bbh":
        return build_bbh_judge_prompt(
            question=judgement_context["question"],
            correct_answer=judgement_context["correct_answer"],
            model_answer=model_answer,
        )
    if dataset_name == "matura":
        return build_matura_judge_prompt(
            task=judgement_context["task"],
            key=judgement_context["klucz"],
            points=judgement_context["punkty"],
            model_answer=model_answer,
        )
    return None


def _save(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(path)


def run_judgement(
    answers_path: Path,
    judge_client: LLMClient,
    output_path: Path,
    existing_judgements: dict[str, dict] | None = None,
) -> bool:
    print(f"\n=== Judgement: loading {answers_path} ===")
    report = json.loads(answers_path.read_text(encoding="utf-8"))
    datasets: dict[str, dict] = report.get("datasets", {})

    judgements: dict[str, dict] = existing_judgements if existing_judgements is not None else {}
    processed_any = False

    for gen_name, gen_datasets in datasets.items():
        if gen_name not in judgements:
            judgements[gen_name] = {}

        for dataset_name, dataset_data in gen_datasets.items():
            records: list[dict] = dataset_data.get("questions", [])
            total = len(records)
            existing_dataset = judgements[gen_name].get(dataset_name)
            existing_questions: list[dict] = []
            if existing_dataset is not None:
                raw_existing_questions = existing_dataset.get("questions", [])
                if isinstance(raw_existing_questions, list):
                    existing_questions = raw_existing_questions

            judged: list[dict] = []
            did_retry = False
            dataset_start = time.perf_counter()

            for i, record in enumerate(records, start=1):
                prompt_id = str(record.get("prompt_id", f"row-{i:03d}"))
                existing_record = (
                    existing_questions[i - 1] if i - 1 < len(existing_questions) else None
                )
                can_reuse = (
                    isinstance(existing_record, dict)
                    and existing_record.get("prompt_id") == prompt_id
                    and not bool(existing_record.get("judge_error", False))
                )
                if can_reuse:
                    judged.append(existing_record)
                    continue

                if not did_retry:
                    print(f"\n  [{gen_name}/{dataset_name}] Judging ...")
                did_retry = True
                processed_any = True

                model_answer = record.get("raw_answer", "")
                judgement_context = record.get("judgement_context", {})

                judge_prompt = _build_judge_prompt(dataset_name, model_answer, judgement_context)
                if judge_prompt is None:
                    current = {**record, "judge_scores": None, "judge_error": True}
                else:
                    t0 = time.perf_counter()
                    try:
                        result = judge_client.ask(judge_prompt, json_mode=True)
                        elapsed = time.perf_counter() - t0
                        print(f"    [{i}/{total}] {result}  ({elapsed:.2f}s)")
                        current = {**record, "judge_scores": result, "judge_error": False}
                    except Exception as e:
                        print(f"    [{i}/{total}] Judge error: {e}")
                        current = {**record, "judge_scores": None, "judge_error": True}

                judged.append(current)
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

            if not did_retry:
                judgements[gen_name][dataset_name] = {
                    **{k: v for k, v in dataset_data.items() if k != "questions"},
                    "judge_error_count": sum(1 for r in judged if r.get("judge_error", False)),
                    "questions": judged,
                }
                print(f"\n  [{gen_name}/{dataset_name}] Already completed, skipping (resume).")
                continue

            error_count = sum(1 for r in judged if r["judge_error"])
            judgements[gen_name][dataset_name] = {
                **{k: v for k, v in dataset_data.items() if k != "questions"},
                "judge_error_count": error_count,
                "questions": judged,
            }
            dataset_elapsed = time.perf_counter() - dataset_start
            print(f"  [{gen_name}/{dataset_name}] judged in {dataset_elapsed:.1f}s")

    if not processed_any and existing_judgements is not None:
        return False

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
    return True


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

    timestamp = datetime.now(POLISH_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = answers_path.parent.name
    output_path = judgements_dir / model_dir / f"{answers_path.stem}_judged_{timestamp}.json"
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

    run_judgement(answers_path, judge_client, output_path, existing_judgements)


if __name__ == "__main__":
    main()