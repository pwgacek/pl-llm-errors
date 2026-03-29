from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def _iter_report_paths(args: list[str]) -> list[Path]:
    if len(args) == 1:
        only = Path(args[0])
        if only.is_dir():
            paths = sorted(path for path in only.glob("*.json") if path.is_file())
            if len(paths) < 2:
                raise ValueError(
                    f"Directory '{only}' must contain at least 2 report files (*.json)."
                )
            return paths

    paths = [Path(arg) for arg in args]
    if len(paths) < 2:
        raise ValueError("Provide at least 2 report files, or one directory with reports.")
    return paths


def _load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _question_key(prompt_id: str, fallback_index: int) -> str:
    match = re.search(r"-(\d+)$", prompt_id)
    if match:
        return match.group(1)
    return f"idx-{fallback_index:04d}"


def _collect_single_report_scores(report: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Return mapping: dataset -> generator -> question_key -> score."""
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    datasets = report.get("datasets", {})

    for generator_name, generator_data in datasets.items():
        if not isinstance(generator_data, dict):
            continue
        for dataset_name, dataset_summary in generator_data.items():
            if not isinstance(dataset_summary, dict):
                continue
            questions = dataset_summary.get("questions", [])
            if not isinstance(questions, list):
                continue

            dataset_scores = grouped.setdefault(dataset_name, {})
            generator_scores: dict[str, float] = {}
            for idx, question in enumerate(questions):
                if not isinstance(question, dict):
                    continue
                prompt_id = str(question.get("prompt_id", "")).strip()
                key = _question_key(prompt_id, idx)
                generator_scores[key] = float(question.get("score", 0.0))

            dataset_scores[generator_name] = generator_scores

    return grouped


def _collect_single_report_raw_answers(report: dict) -> dict[str, dict[str, dict[str, str]]]:
    """Return mapping: dataset -> generator -> question_key -> raw_answer_text."""
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    datasets = report.get("datasets", {})

    for generator_name, generator_data in datasets.items():
        if not isinstance(generator_data, dict):
            continue
        for dataset_name, dataset_summary in generator_data.items():
            if not isinstance(dataset_summary, dict):
                continue
            questions = dataset_summary.get("questions", [])
            if not isinstance(questions, list):
                continue

            dataset_answers = grouped.setdefault(dataset_name, {})
            generator_answers: dict[str, str] = {}
            for idx, question in enumerate(questions):
                if not isinstance(question, dict):
                    continue
                prompt_id = str(question.get("prompt_id", "")).strip()
                key = _question_key(prompt_id, idx)
                generator_answers[key] = str(question.get("raw_answer", ""))

            dataset_answers[generator_name] = generator_answers

    return grouped


def _generator_permutations() -> dict[str, dict[str, list[int]]]:
    from src.prompt_preparation.prompt_builder import GENERATORS

    result: dict[str, dict[str, list[int]]] = {}
    for generator_name, generator in GENERATORS.items():
        raw = getattr(generator, "answer_permutations", {}) or {}
        if not isinstance(raw, dict):
            continue

        normalized: dict[str, list[int]] = {}
        for dataset_name, permutation in raw.items():
            if not isinstance(dataset_name, str):
                continue
            if not isinstance(permutation, (list, tuple)):
                continue
            try:
                normalized[dataset_name] = [int(value) for value in permutation]
            except (TypeError, ValueError):
                continue
        result[generator_name] = normalized
    return result


def _extract_answer_letter(raw_answer: str) -> str | None:
    if not raw_answer:
        return None

    try:
        parsed = json.loads(raw_answer)
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, str):
                    match = re.search(r"[A-Za-z]", value)
                    if match:
                        return match.group(0).upper()
    except json.JSONDecodeError:
        pass

    match = re.search(r'"([A-Za-z])"', raw_answer)
    if match:
        return match.group(1).upper()

    return None


def _canonical_answer_token(
    letter: str | None,
    permutation: list[int] | None,
    option_count: int | None,
) -> str:
    if letter is None:
        return "missing"

    idx = ord(letter.upper()) - ord("A")
    if idx < 0:
        return f"invalid:{letter}"

    if option_count is not None and idx >= option_count:
        return f"invalid:{letter}"

    if permutation is None:
        return str(idx)

    if idx >= len(permutation):
        return f"invalid:{letter}"

    return str(permutation[idx])


def _dataset_option_count(
    dataset_name: str,
    permutations_by_generator: dict[str, dict[str, list[int]]],
) -> int | None:
    for generator_permutations in permutations_by_generator.values():
        permutation = generator_permutations.get(dataset_name)
        if permutation is not None:
            return len(permutation)

    defaults = {"llmzszl": 4, "bbh": 7}
    return defaults.get(dataset_name)


def _dataset_stability(
    dataset_name: str,
    raw_answers_by_generator: dict[str, dict[str, str]],
    generators: list[str],
    permutations_by_generator: dict[str, dict[str, list[int]]],
) -> tuple[float, int]:
    if not generators:
        return 0.0, 0

    option_count = _dataset_option_count(dataset_name, permutations_by_generator)

    question_keys = set(raw_answers_by_generator.get(generators[0], {}).keys())
    for generator_name in generators[1:]:
        question_keys &= set(raw_answers_by_generator.get(generator_name, {}).keys())

    if not question_keys:
        return 0.0, 0

    stability_sum = 0.0
    for question_key in sorted(question_keys):
        canonical_answers: list[str] = []
        for generator_name in generators:
            raw_answer = raw_answers_by_generator[generator_name].get(question_key, "")
            letter = _extract_answer_letter(raw_answer)
            permutation = permutations_by_generator.get(generator_name, {}).get(dataset_name)
            canonical = _canonical_answer_token(letter, permutation, option_count)
            canonical_answers.append(canonical)

        max_count = Counter(canonical_answers).most_common(1)[0][1]
        stability_sum += max_count / len(generators)

    return stability_sum / len(question_keys), len(question_keys)


def _collect_scores(report: dict) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    datasets = report.get("datasets", {})

    for generator_name, generator_data in datasets.items():
        if not isinstance(generator_data, dict):
            continue
        for dataset_name, dataset_summary in generator_data.items():
            if not isinstance(dataset_summary, dict):
                continue
            questions = dataset_summary.get("questions", [])
            if not isinstance(questions, list):
                continue
            for question in questions:
                if not isinstance(question, dict):
                    continue
                prompt_id = str(question.get("prompt_id", "")).strip()
                if not prompt_id:
                    continue
                score = float(question.get("score", 0.0))
                scores[(generator_name, dataset_name, prompt_id)] = score

    return scores


def _correct(score: float) -> bool:
    return score > 0.0


def _accuracy(scores: dict[tuple[str, str, str], float], keys: set[tuple[str, str, str]]) -> float:
    if not keys:
        return 0.0
    correct_count = sum(1 for key in keys if _correct(scores[key]))
    return correct_count / len(keys)


def _compare_pair(
    baseline_scores: dict[tuple[str, str, str], float],
    candidate_scores: dict[tuple[str, str, str], float],
) -> dict[str, int | float]:
    baseline_keys = set(baseline_scores.keys())
    candidate_keys = set(candidate_scores.keys())
    common_keys = baseline_keys & candidate_keys

    c_to_c = 0
    c_to_i = 0
    i_to_c = 0
    i_to_i = 0

    for key in common_keys:
        baseline_correct = _correct(baseline_scores[key])
        candidate_correct = _correct(candidate_scores[key])

        if baseline_correct and candidate_correct:
            c_to_c += 1
        elif baseline_correct and not candidate_correct:
            c_to_i += 1
        elif not baseline_correct and candidate_correct:
            i_to_c += 1
        else:
            i_to_i += 1

    total = c_to_c + c_to_i + i_to_c + i_to_i

    return {
        "baseline_only": len(baseline_keys - candidate_keys),
        "candidate_only": len(candidate_keys - baseline_keys),
        "compared": total,
        "baseline_accuracy": _accuracy(baseline_scores, common_keys),
        "candidate_accuracy": _accuracy(candidate_scores, common_keys),
        "correct_to_correct": c_to_c,
        "correct_to_incorrect": c_to_i,
        "incorrect_to_correct": i_to_c,
        "incorrect_to_incorrect": i_to_i,
        "sum_of_all_categories": total,
        "changed_correctness_total": c_to_i + i_to_c,
    }


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _compare_generators_within_report(
    report_path: Path,
    report: dict,
    baseline_generator: str = "identity",
) -> None:
    grouped = _collect_single_report_scores(report)
    raw_answers_grouped = _collect_single_report_raw_answers(report)
    permutations_by_generator = _generator_permutations()

    print(f"Report: {report_path}")
    print(f"Baseline generator: {baseline_generator}")
    print()

    for dataset_name in sorted(grouped.keys()):
        generator_scores = grouped[dataset_name]
        if baseline_generator not in generator_scores:
            print(f"Dataset: {dataset_name}")
            print(f"  Missing baseline generator '{baseline_generator}' in this dataset.")
            print()
            continue

        baseline_scores = generator_scores[baseline_generator]
        candidate_names = sorted(
            name for name in generator_scores.keys() if name != baseline_generator
        )
        compared_generators = [baseline_generator, *candidate_names]
        stability, stability_questions = _dataset_stability(
            dataset_name=dataset_name,
            raw_answers_by_generator=raw_answers_grouped.get(dataset_name, {}),
            generators=compared_generators,
            permutations_by_generator=permutations_by_generator,
        )

        print(f"Dataset: {dataset_name}")
        print(
            f"  Stability: {stability:.4f} (avg max-frequency over {stability_questions} questions)"
        )
        if not candidate_names:
            print("  No candidate generators to compare.")
            print()
            continue

        for candidate_name in candidate_names:
            candidate_scores = generator_scores[candidate_name]
            comparison = _compare_pair(baseline_scores, candidate_scores)

            print(f"  Compare {baseline_generator} vs {candidate_name}")
            print(f"    Compared questions: {comparison['compared']}")
            print(
                f"    Baseline accuracy: {_format_percent(float(comparison['baseline_accuracy']))}"
            )
            print(
                f"    Candidate accuracy: {_format_percent(float(comparison['candidate_accuracy']))}"
            )
            print(f"    Correct -> Correct: {comparison['correct_to_correct']}")
            print(f"    Correct -> Incorrect: {comparison['correct_to_incorrect']}")
            print(f"    Incorrect -> Correct: {comparison['incorrect_to_correct']}")
            print(f"    Incorrect -> Incorrect: {comparison['incorrect_to_incorrect']}")
            print(f"    Changed correctness total: {comparison['changed_correctness_total']}")
        print()


def main(argv: list[str]) -> None:
    if len(argv) == 1 and Path(argv[0]).is_file():
        report_path = Path(argv[0])
        report = _load_report(report_path)
        _compare_generators_within_report(report_path, report, baseline_generator="identity")
        return

    report_paths = _iter_report_paths(argv)
    reports = [_load_report(path) for path in report_paths]
    score_maps = [_collect_scores(report) for report in reports]

    baseline_path = report_paths[0]
    baseline_scores = score_maps[0]

    print(f"Baseline: {baseline_path}")
    print()

    for index in range(1, len(report_paths)):
        candidate_path = report_paths[index]
        candidate_scores = score_maps[index]
        comparison = _compare_pair(baseline_scores, candidate_scores)

        print(f"Compare baseline vs run {index + 1}: {candidate_path}")
        print(f"  Compared questions: {comparison['compared']}")
        print(f"  Baseline accuracy: {_format_percent(float(comparison['baseline_accuracy']))}")
        print(f"  Candidate accuracy: {_format_percent(float(comparison['candidate_accuracy']))}")
        print(f"  Correct -> Correct: {comparison['correct_to_correct']}")
        print(f"  Correct -> Incorrect: {comparison['correct_to_incorrect']}")
        print(f"  Incorrect -> Correct: {comparison['incorrect_to_correct']}")
        print(f"  Incorrect -> Incorrect: {comparison['incorrect_to_incorrect']}")
        print(f"  Changed correctness total: {comparison['changed_correctness_total']}")
        print()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as error:
        print(f"Error: {error}")
        print(
            "Usage: python -m src.scripts.compare_runs <baseline_report.json> <other_report.json> [more_reports...]"
        )
        print(
            "   or: python -m src.scripts.compare_runs <single_report.json>  # identity vs permutation_* per dataset"
        )
        print("   or: python -m src.scripts.compare_runs <reports_directory>")
        raise SystemExit(1)
