from __future__ import annotations

import json
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from settings import settings
from questions import (
    BBHQuestion,
    CdsQuestion,
    LDEKQuestion,
    LlmzszlQuestion,
    PolQAQuestion,
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
        "url": "http://git.nlp.ipipan.waw.pl/Scwad/SCWAD-CDSCorpus/raw/master/CDSCorpus/CDS_test.csv",
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

def _download_file(url: str, output: Path, timeout: int = 120) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with output.open("wb") as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)


def _serialize_expected(question: Question) -> dict:
    """Return a minimal, serialisable dict describing the correct answer."""
    if isinstance(question, LlmzszlQuestion):
        return {"type": "multiple_choice_index", "correct_index": question.correct_answer_index}
    if isinstance(question, LDEKQuestion):
        return {"type": "multiple_choice_letter", "correct_letter": question.correct_answer}
    if isinstance(question, PolQAQuestion):
        return {"type": "open_contained", "accepted_answers": question.answers}
    if isinstance(question, CdsQuestion):
        return {"type": "entailment", "judgment": question.entailment_judgment}
    if isinstance(question, BBHQuestion):
        return {"type": "multiple_choice_letter", "correct_letter": question.answer}
    raise TypeError(f"Unknown question type: {type(question)}")


def _download_datasets() -> None:
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
            _download_file(url, output)
            print(f"  [{name}] Done.")
            downloaded += 1
        except urllib.error.URLError as e:
            print(f"  [{name}] FAILED: {e}")
            failed += 1
    print(f"  downloaded={downloaded}  skipped={skipped}  failed={failed}")
    if failed:
        print("  Some downloads failed. Proceeding with available datasets.")


def _load_datasets(num_questions: int, seed: int) -> dict[str, list[Question]]:
    print("\n=== Step 2: Loading datasets ===")
    loaded: dict[str, list[Question]] = {}
    for dataset in DATASETS:
        name = dataset["name"]
        if not dataset["output"].exists():
            print(f"  [{name}] File missing, skipping.")
            continue
        try:
            questions = dataset["loader"]().load(
                path=dataset["output"], num_samples=num_questions, seed=seed
            )
            loaded[name] = questions
            print(f"  [{name}] Loaded {len(questions)} questions.")
        except Exception as e:
            print(f"  [{name}] Load error: {e}")
    return loaded


def _clean_output_directory(output_dir: Path) -> None:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return

    print(f"  [info] Cleaning output directory: {output_dir}")
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _build_prompt_lines(
    questions: list[Question],
    dataset_name: str,
    gen_name: str,
    generator: ErrorGenerator,
) -> list[str]:
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
    return lines


def _save_dataset_prompts(
    dataset_name: str,
    questions: list[Question],
    gen_name: str,
    generator: ErrorGenerator,
    gen_dir: Path,
) -> None:
    out_path = gen_dir / f"{dataset_name}.jsonl"
    if out_path.exists():
        out_path.unlink()
        print(f"  [{gen_name}/{dataset_name}] Overwriting existing file.")

    lines = _build_prompt_lines(
        questions=questions,
        dataset_name=dataset_name,
        gen_name=gen_name,
        generator=generator,
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [{gen_name}/{dataset_name}] Saved {len(questions)} prompts to {out_path}")


def _build_and_save_prompts(loaded: dict[str, list[Question]], output_dir: Path) -> None:
    print("\n=== Step 3: Building and saving noised prompts ===")
    _clean_output_directory(output_dir)

    for gen_name, generator in GENERATORS.items():
        gen_dir = output_dir / gen_name
        gen_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, questions in loaded.items():
            _save_dataset_prompts(
                dataset_name=dataset_name,
                questions=questions,
                gen_name=gen_name,
                generator=generator,
                gen_dir=gen_dir,
            )


def build_prompts(
    num_questions: int,
    seed: int,
    output_dir: Path,
) -> None:
    _download_datasets()

    loaded = _load_datasets(num_questions=num_questions, seed=seed)
    if not loaded:
        print("No datasets loaded. Nothing to build.")
        return

    _build_and_save_prompts(loaded=loaded, output_dir=output_dir)
    print("\nDone. Prompts saved to:", output_dir)


def main() -> None:
    build_prompts(
        num_questions=settings.prompt_preparation.num_questions,
        seed=settings.prompt_preparation.seed,
        output_dir=Path(settings.prompt_preparation.output_dir),
    )


if __name__ == "__main__":
    main()
