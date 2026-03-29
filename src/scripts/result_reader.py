import json
from collections import defaultdict
from pathlib import Path
import sys


def _extract_counts(summary: dict) -> tuple[float, int]:
	total_raw = summary.get("num_sampled", 0)
	total = int(total_raw) if isinstance(total_raw, (int, float)) else 0

	if "score_sum" in summary:
		score_sum_raw = summary.get("score_sum", 0.0)
		correct = float(score_sum_raw) if isinstance(score_sum_raw, (int, float)) else 0.0
		if total == 0:
			questions = summary.get("questions", [])
			if isinstance(questions, list):
				total = len(questions)
		return correct, total

	if "correct" in summary:
		correct_raw = summary.get("correct", 0)
		correct = float(correct_raw) if isinstance(correct_raw, (int, float)) else 0.0
		if total == 0:
			questions = summary.get("questions", [])
			if isinstance(questions, list):
				total = len(questions)
		return correct, total

	questions = summary.get("questions", [])
	if isinstance(questions, list):
		correct = sum(
			1.0
			for question in questions
			if isinstance(question, dict)
			and isinstance(question.get("score", 0), (int, float))
			and float(question.get("score", 0)) > 0.0
		)
		if total == 0:
			total = len(questions)
		return correct, total

	return 0.0, total


def main(report_path: str) -> None:
	report = json.loads(Path(report_path).read_text(encoding="utf-8"))

	data = report.get("datasets", {})
	if not isinstance(data, dict):
		raise ValueError("Invalid report format: 'datasets' must be a dictionary")

	error_type_acc: defaultdict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
	pair_acc: defaultdict[tuple[str, str], list[float | int]] = defaultdict(lambda: [0.0, 0])

	for error_type, datasets in data.items():
		if not isinstance(error_type, str) or not isinstance(datasets, dict):
			continue
		for dataset, summary in datasets.items():
			if not isinstance(dataset, str) or not isinstance(summary, dict):
				continue
			correct, total = _extract_counts(summary)
			error_type_acc[error_type][0] += correct
			error_type_acc[error_type][1] += total
			pair_acc[(dataset, error_type)][0] += correct
			pair_acc[(dataset, error_type)][1] += total

	datasets = sorted({dataset for dataset, _ in pair_acc})
	error_types = sorted({error_type for _, error_type in pair_acc})

	print("Accuracy per dataset (all error types):")
	for dataset in datasets:
		print(f"{dataset}:")
		for error_type in error_types:
			corr, tot = pair_acc.get((dataset, error_type), (0.0, 0))
			acc = float(corr) / int(tot) if int(tot) else 0.0
			print(f"  {error_type}: {acc:.3f} ({float(corr):.1f}/{int(tot)})")

	print("\nAccuracy per error type (all datasets):")
	for error_type in error_types:
		corr, tot = error_type_acc[error_type]
		acc = float(corr) / int(tot) if int(tot) else 0.0
		print(f"  {error_type}: {acc:.3f} ({float(corr):.1f}/{int(tot)})")

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python -m src.scripts.result_reader <report_path>")
	else:
		main(sys.argv[1])
