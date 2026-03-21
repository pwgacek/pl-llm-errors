from __future__ import annotations

import argparse
import json
import urllib.error
from pathlib import Path

from download import download_file
from errors import (
    DiacriticErrorGenerator,
    IdentityGenerator,
    PunctuationAllErrorGenerator,
    PunctuationInnerErrorGenerator,
    SpellingErrorGenerator,
    TypoErrorGenerator,
)
from errors.base import ErrorGenerator
from loaders import CDSLoader, LDEKLoader, LLMZSZLLoader, PolQALoader, BBHLoader
from questions import (
    BBHQuestion,
    CdsQuestion,
    LDEKQuestion,
    LlmzszlQuestion,
    PolqaQuestion,
    Question,
)

GENERATORS: dict[str, ErrorGenerator] = {
    "identity": IdentityGenerator(),
    "diacritic": DiacriticErrorGenerator(),
    "punctuation_all": PunctuationAllErrorGenerator(),
    "punctuation_inner": PunctuationInnerErrorGenerator(),
    "spelling_10%": SpellingErrorGenerator(rate=0.1),
    "spelling_40%": SpellingErrorGenerator(rate=0.4),
    "typo_10%": TypoErrorGenerator(typo_rate=0.1),
    "typo_40%": TypoErrorGenerator(typo_rate=0.4),
}


DATASETS = [
    {
        "name": "llmzszl",
        "url": "https://huggingface.co/datasets/amu-cai/llmzszl-dataset/resolve/main/llmzszl-test.jsonl",
        "output": Path("datasets/llmzszl.jsonl"),
        "loader": LLMZSZLLoader,
    },
    {
        "name": "polqa",
        "url": "https://huggingface.co/datasets/ipipan/polqa/resolve/main/data/test.csv",
        "output": Path("datasets/polqa.csv"),
        "loader": PolQALoader,
    },
    {
        "name": "cds",
        "url": "https://git.nlp.ipipan.waw.pl/Scwad/SCWAD-CDSCorpus/repository/archive.zip",
        "output": Path("datasets/CDS_test.csv"),
        "loader": CDSLoader,
    },
    {
        "name": "ldek",
        "url": "https://huggingface.co/datasets/amu-cai/medical-exams-LDEK-PL-2008-2024/resolve/main/medical-exams-LDEK-PL-2008-2024.json",
        "output": Path("datasets/medical-exams-LDEK-PL-2008-2024.json"),
        "loader": LDEKLoader,
    },
    {
        "name": "bbh",
        "url": "https://huggingface.co/datasets/pawel04/bbh-logical-deduction-seven-objects-pl/resolve/main/data.jsonl",
        "output": Path("datasets/bbh-logical-deduction-seven-objects-pl.jsonl"),
        "loader": BBHLoader,
    },
]


def _serialize_expected(question: Question) -> dict:
    """Return a minimal, serialisable dict describing the correct answer."""
    if isinstance(question, LlmzszlQuestion):
        return {"type": "multiple_choice_index", "correct_index": question.correct_answer_index}
    if isinstance(question, LDEKQuestion):
        return {"type": "multiple_choice_letter", "correct_letter": question.correct_answer}
    if isinstance(question, PolqaQuestion):
        return {"type": "open_contained", "accepted_answers": question.answers}
    if isinstance(question, CdsQuestion):
        return {"type": "entailment", "judgment": question.entailment_judgment}
    if isinstance(question, BBHQuestion):
        return {"type": "multiple_choice_letter", "correct_letter": question.answer}
    raise TypeError(f"Unknown question type: {type(question)}")


class PromptBuilder:
    def __init__(
        self,
        num_questions: int = 100,
        seed: int = 42,
        output_dir: Path = Path("prompts"),
        skip_download: bool = False,
    ) -> None:
        self.num_questions = num_questions
        self.seed = seed
        self.output_dir = output_dir
        self.skip_download = skip_download

    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------

    def download(self) -> None:
        print("\n=== Step 1: Downloading datasets ===")
        downloaded = skipped = failed = 0
        for dataset in DATASETS:
            name, url, output = dataset["name"], dataset["url"], dataset["output"]
            if output.exists():
                print(f"  [{name}] Already exists, skipping.")
                skipped += 1
                continue
            print(f"  [{name}] Downloading {url} ...")
            try:
                download_file(url, output)
                print(f"  [{name}] Done.")
                downloaded += 1
            except urllib.error.URLError as e:
                print(f"  [{name}] FAILED: {e}")
                failed += 1
        print(f"  downloaded={downloaded}  skipped={skipped}  failed={failed}")
        if failed:
            print("  Some downloads failed. Proceeding with available datasets.")

    # ------------------------------------------------------------------
    # Step 2: Load
    # ------------------------------------------------------------------

    def load(self) -> dict[str, list[Question]]:
        print("\n=== Step 2: Loading datasets ===")
        loaded: dict[str, list[Question]] = {}
        for dataset in DATASETS:
            name = dataset["name"]
            if not dataset["output"].exists():
                print(f"  [{name}] File missing, skipping.")
                continue
            try:
                questions = dataset["loader"]().load(num_samples=self.num_questions, seed=self.seed)
                loaded[name] = questions
                print(f"  [{name}] Loaded {len(questions)} questions.")
            except Exception as e:
                print(f"  [{name}] Load error: {e}")
        return loaded

    # ------------------------------------------------------------------
    # Step 3: Build & save prompts
    # ------------------------------------------------------------------

    def build_and_save(self, loaded: dict[str, list[Question]]) -> None:
        print("\n=== Step 3: Building and saving noised prompts ===")
        # Clean output_dir if it exists and is not empty
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            print(f"  [info] Cleaning output directory: {self.output_dir}")
            import shutil
            for item in self.output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        for gen_name, generator in GENERATORS.items():
            gen_dir = self.output_dir / gen_name
            gen_dir.mkdir(parents=True, exist_ok=True)

            for dataset_name, questions in loaded.items():
                out_path = gen_dir / f"{dataset_name}.jsonl"
                if out_path.exists():
                    out_path.unlink()
                    print(f"  [{gen_name}/{dataset_name}] Overwriting existing file.")

                lines: list[str] = []
                for idx, question in enumerate(questions, start=0):
                    prompt = question.build_prompt(generator)
                    expected = _serialize_expected(question)
                    record = {
                        "id": f"{dataset_name}-{gen_name}-{idx:03d}",
                        "prompt": prompt,
                        "expected": expected,
                    }
                    lines.append(json.dumps(record, ensure_ascii=False))

                out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                print(f"  [{gen_name}/{dataset_name}] Saved {len(questions)} prompts to {out_path}")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def build(self) -> None:
        if not self.skip_download:
            self.download()
        loaded = self.load()
        if not loaded:
            print("No datasets loaded. Nothing to build.")
            return
        self.build_and_save(loaded)
        print("\nDone. Prompts saved to:", self.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline pipeline: download → load → noisify → save prompts."
    )
    parser.add_argument("--num-questions", type=int, default=100, help="Questions sampled per dataset (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output-dir", default="prompts", help="Root directory for saved prompts (default: prompts/).")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download step.")
    args = parser.parse_args()

    PromptBuilder(
        num_questions=args.num_questions,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        skip_download=args.skip_download,
    ).build()


if __name__ == "__main__":
    main()
