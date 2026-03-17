import json
from collections import defaultdict

def main(report_path):
	with open(report_path, 'r', encoding='utf-8') as f:
		report = json.load(f)

	data = report.get('datasets', {})
	error_type_acc = defaultdict(lambda: [0, 0])  # correct, total
	pair_acc = defaultdict(lambda: [0, 0])

	for error_type, datasets in data.items():
		for dataset, summary in datasets.items():
			correct = summary.get('correct', 0)
			total = summary.get('num_sampled', 0)
			error_type_acc[error_type][0] += correct
			error_type_acc[error_type][1] += total
			pair_acc[(dataset, error_type)][0] += correct
			pair_acc[(dataset, error_type)][1] += total

	# Gather all datasets and error types
	datasets = set()
	error_types = set()
	for (ds, et) in pair_acc:
		datasets.add(ds)
		error_types.add(et)
	datasets = sorted(datasets)
	error_types = sorted(error_types)

	print("Accuracy per dataset (all error types):")
	for ds in datasets:
		print(f"{ds}:")
		for et in error_types:
			corr, tot = pair_acc.get((ds, et), (0, 0))
			acc = corr / tot if tot else 0
			print(f"  {et}: {acc:.3f} ({corr}/{tot})")

if __name__ == "__main__":
	import sys
	if len(sys.argv) < 2:
		print("Usage: python result_reader.py <report_path>")
	else:
		main(sys.argv[1])
