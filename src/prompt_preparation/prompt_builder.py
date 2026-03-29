from __future__ import annotations

import json
import shutil
import urllib.request
import urllib.error
from pathlib import Path

from .errors import (
    DiacriticErrorGenerator,
    IdentityGenerator,
    PunctuationAllErrorGenerator,
    PunctuationInnerErrorGenerator,
    SpellingErrorGenerator,
    TypoErrorGenerator,
)
from .errors.base import ErrorGenerator
from .loaders import CDSLoader, LDEKLoader, LLMZSZLLoader, PolQALoader, BBHLoader
from .questions import (
    BBHQuestion,
    CdsQuestion,
    LDEKQuestion,
    LlmzszlQuestion,
    PolQAQuestion,
    Question,
)

from src.settings import settings

GENERATORS: dict[str, ErrorGenerator] = {
    "identity": IdentityGenerator(),
    "temp_1": IdentityGenerator(
        temperature=0.1,
    ),
    "temp_2": IdentityGenerator(
        temperature=0.3,
    ),
    "temp_3": IdentityGenerator(
        temperature=0.5,
    ),
}


DATASETS = [
    {
        "name": "llmzszl",
        "url": "https://huggingface.co/datasets/amu-cai/llmzszl-dataset/resolve/main/llmzszl-test.jsonl",
        "output": Path("datasets/llmzszl.jsonl"),
        "loader": LLMZSZLLoader,
    },
    {
        "name": "bbh",
        "url": "https://huggingface.co/datasets/pawel04/bbh-logical-deduction-seven-objects-pl/resolve/main/data.jsonl",
        "output": Path("datasets/bbh-logical-deduction-seven-objects-pl.jsonl"),
        "loader": BBHLoader,
    },
]


def _generator_answer_permutations(generator: ErrorGenerator) -> dict[str, list[int]]:
    raw = getattr(generator, "answer_permutations", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("IdentityGenerator.answer_permutations must be a dictionary")

    normalized: dict[str, list[int]] = {}
    for dataset_name, permutation in raw.items():
        if not isinstance(dataset_name, str):
            raise ValueError("Permutation dataset name must be a string")
        if not isinstance(permutation, (list, tuple)):
            raise ValueError(
                f"Permutation for dataset '{dataset_name}' must be a list of integers"
            )
        try:
            normalized[dataset_name] = [int(value) for value in permutation]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Permutation for dataset '{dataset_name}' contains a non-integer value"
            ) from error
    return normalized


def _generator_temperature(generator: ErrorGenerator) -> float | None:
    raw = getattr(generator, "temperature", None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("Generator temperature must be a number or None") from error


def _permuted_question(
    question: Question,
    dataset_name: str,
    answer_permutations: dict[str, list[int]],
) -> Question:
    permutation = answer_permutations.get(dataset_name)
    if permutation is None:
        return question

    if isinstance(question, LlmzszlQuestion):
        expected = list(range(len(question.answers)))
        if sorted(permutation) != expected:
            raise ValueError(
                f"Invalid llmzszl permutation={permutation}, expected permutation of {expected}"
            )
        answers = [question.answers[idx] for idx in permutation]
        correct_answer_index = permutation.index(question.correct_answer_index)
        return LlmzszlQuestion(question.question, answers, correct_answer_index)

    if isinstance(question, BBHQuestion):
        expected = list(range(len(question.options)))
        if sorted(permutation) != expected:
            raise ValueError(
                f"Invalid bbh permutation={permutation}, expected permutation of {expected}"
            )
        options = [question.options[idx] for idx in permutation]
        correct_index = ord(question.answer.upper()) - ord("A")
        if correct_index < 0 or correct_index >= len(question.options):
            raise ValueError(f"Invalid BBH correct answer letter: {question.answer}")
        remapped_correct_index = permutation.index(correct_index)
        answer = chr(ord("A") + remapped_correct_index)
        return BBHQuestion(question.text, options, answer)

    raise ValueError(
        f"Answer permutation configured for unsupported dataset '{dataset_name}' and question type '{type(question).__name__}'"
    )

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
        return {"type": "open_short_answer", "accepted_answers": question.answers}
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
    answer_permutations: dict[str, list[int]],
    generator_temperature: float | None,
) -> list[str]:
    lines: list[str] = []
    for idx, question in enumerate(questions, start=0):
        effective_question = _permuted_question(question, dataset_name, answer_permutations)
        prompt = effective_question.build_prompt(generator)
        expected = _serialize_expected(effective_question)
        record = {
            "id": f"{dataset_name}-{gen_name}-{idx:03d}",
            "prompt": prompt,
            "expected": expected,
        }
        if generator_temperature is not None:
            record["temperature"] = generator_temperature
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


def _save_dataset_prompts(
    dataset_name: str,
    questions: list[Question],
    gen_name: str,
    generator: ErrorGenerator,
    gen_dir: Path,
    answer_permutations: dict[str, list[int]],
    generator_temperature: float | None,
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
        answer_permutations=answer_permutations,
        generator_temperature=generator_temperature,
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [{gen_name}/{dataset_name}] Saved {len(questions)} prompts to {out_path}")


def _build_and_save_prompts(loaded: dict[str, list[Question]], output_dir: Path) -> None:
    print("\n=== Step 3: Building and saving noised prompts ===")
    _clean_output_directory(output_dir)

    for gen_name, generator in GENERATORS.items():
        answer_permutations = _generator_answer_permutations(generator)
        generator_temperature = _generator_temperature(generator)
        gen_dir = output_dir / gen_name
        gen_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, questions in loaded.items():
            _save_dataset_prompts(
                dataset_name=dataset_name,
                questions=questions,
                gen_name=gen_name,
                generator=generator,
                gen_dir=gen_dir,
                answer_permutations=answer_permutations,
                generator_temperature=generator_temperature,
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
        seed=settings.common.seed,
        output_dir=Path(settings.common.prompt_dir),
    )


if __name__ == "__main__":
    main()
