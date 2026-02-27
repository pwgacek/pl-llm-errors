"""Extract orthographic (ort) errors from ALL JSONL files in the
polish-gec-datasets folder and save as a correct → incorrect JSON dictionary."""

import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "polish-gec-datasets"
DST = Path(__file__).parent.parent / "assets" / "polish_gec_dataset_spelling_errors.json"


def main() -> None:
    mapping: dict[str, str] = {}

    jsonl_files = sorted(SRC_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {SRC_DIR}")
        return

    for src in jsonl_files:
        count_before = len(mapping)
        with open(src, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for err in record.get("errors", []):
                    if err.get("type") != "ort":
                        continue
                    correct = err["correct"].strip().lower()
                    incorrect = err["incorrect"].strip().lower()
                    # skip if only capitalization differs or strings are equal
                    if not correct or not incorrect or correct == incorrect:
                        continue
                    # keep first occurrence (no overwrite)
                    if correct not in mapping:
                        mapping[correct] = incorrect
        added = len(mapping) - count_before
        print(f"  {src.name}: +{added} new entries")

    # sort alphabetically for readability
    sorted_mapping = dict(sorted(mapping.items()))

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(sorted_mapping, f, ensure_ascii=False, indent=4)

    print(f"\nExtracted {len(sorted_mapping)} unique ort errors → {DST.name}")


if __name__ == "__main__":
    main()
