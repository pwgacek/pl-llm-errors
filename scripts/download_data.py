from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT_SECONDS = 120

DATASETS = [
    {
        "name": "llmzszl",
        "url": "https://huggingface.co/datasets/amu-cai/llmzszl-dataset/resolve/main/llmzszl-test.jsonl",
        "output": Path("datasets/llmzszl.jsonl"),
    },
    {
        "name": "belebele-pol",
        "url": "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/pol_Latn.jsonl",
        "output": Path("datasets/belebele-pol.jsonl"),
    },
    {
        "name": "polqa",
        "url": "https://huggingface.co/datasets/ipipan/polqa/resolve/main/data/test.csv",
        "output": Path("datasets/polqa.csv"),
    },
]

def download_file(url: str, output: Path, timeout: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with output.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)


def main() -> None:
    downloaded = 0
    skipped = 0
    failed = 0

    for dataset in DATASETS:
        name = dataset["name"]
        url = dataset["url"]
        output = dataset["output"]

        if output.exists():
            print(f"[{name}] Skipping (already exists): {output}")
            skipped += 1
            continue

        print(f"[{name}] Downloading: {url}")
        print(f"[{name}] Saving to : {output}")

        try:
            download_file(url, output, TIMEOUT_SECONDS)
            print(f"[{name}] Done")
            downloaded += 1
        except urllib.error.URLError as error:
            print(f"[{name}] Download failed: {error}")
            failed += 1

    print("\nSummary:")
    print(f"downloaded: {downloaded}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
