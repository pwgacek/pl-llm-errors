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
from .loaders import LLMZSZLLoader, PolQALoader, BBHLoader, LLMZSZLMCLoader
from .prompts import (
    BBHPrompt,
    LlmzszlPrompt,
    LlmzszlMCPrompt,
    PolQAPrompt,
    Prompt,
)

from src.settings import settings

GENERATORS: dict[str, ErrorGenerator] = {
    "identity": IdentityGenerator(),
    "typo_40%": TypoErrorGenerator(typo_rate=0.4, seed=settings.common.seed),
}


DATASETS = [
    # {
    #     "name": "llmzszl",
    #     "url": "https://huggingface.co/datasets/pawel04/llmzszl-open-ended/resolve/main/llmzszl-open-ended.jsonl",
    #     "output": Path("datasets/llmzszl-open.jsonl"),
    #     "loader": LLMZSZLLoader,
    # },
    {
        "name": "llmzszl_mc",
        "url": "https://huggingface.co/datasets/pawel04/llmzszl-multiple-choice/resolve/main/llmzszl-multiple-choice.jsonl",
        "output": Path("datasets/llmzszl-mcq.jsonl"),
        "loader": LLMZSZLMCLoader,
    }

]

def _download_file(url: str, output: Path, timeout: int = 120) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with output.open("wb") as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)


def _serialize_judgement_context(prompt_item: Prompt) -> dict:
    """Return prompt source text, optional context, and correct answer for LLM-as-a-judge."""
    if isinstance(prompt_item, LlmzszlPrompt) or isinstance(prompt_item, LlmzszlMCPrompt):
        return {
            "question": prompt_item.question,
            "correct_answer": prompt_item.answers,
        }
    if isinstance(prompt_item, PolQAPrompt):
        return {
            "question": prompt_item.question,
            "context": prompt_item.context,
            "correct_answer": prompt_item.answers,
        }
    if isinstance(prompt_item, BBHPrompt):
        return {
            "question": prompt_item.text,
            "correct_answer": prompt_item.correct_order,
        }
    raise TypeError(f"Unknown prompt type: {type(prompt_item)}")


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


def _load_datasets(num_questions: int, seed: int) -> dict[str, list[Prompt]]:
    print("\n=== Step 2: Loading datasets ===")
    loaded: dict[str, list[Prompt]] = {}
    for dataset in DATASETS:
        name = dataset["name"]
        if not dataset["output"].exists():
            print(f"  [{name}] File missing, skipping.")
            continue
        try:
            prompts = dataset["loader"]().load(
                path=dataset["output"], num_samples=num_questions, seed=seed
            )
            loaded[name] = prompts
            print(f"  [{name}] Loaded {len(prompts)} prompts.")
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
    prompts: list[Prompt],
    dataset_name: str,
    gen_name: str,
    generator: ErrorGenerator,
) -> list[str]:
    lines: list[str] = []
    for idx, prompt_item in enumerate(prompts, start=0):
        prompt = prompt_item.build_prompt(generator)
        record = {
            "id": f"{dataset_name}-{gen_name}-{idx:03d}",
            "prompt": prompt,
            "judgement_context": _serialize_judgement_context(prompt_item),
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


def _save_dataset_prompts(
    dataset_name: str,
    prompts: list[Prompt],
    gen_name: str,
    generator: ErrorGenerator,
    gen_dir: Path,
) -> None:
    out_path = gen_dir / f"{dataset_name}.jsonl"
    if out_path.exists():
        out_path.unlink()
        print(f"  [{gen_name}/{dataset_name}] Overwriting existing file.")

    lines = _build_prompt_lines(
        prompts=prompts,
        dataset_name=dataset_name,
        gen_name=gen_name,
        generator=generator,
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [{gen_name}/{dataset_name}] Saved {len(prompts)} prompts to {out_path}")


def _build_and_save_prompts(loaded: dict[str, list[Prompt]], output_dir: Path) -> None:
    print("\n=== Step 3: Building and saving noised prompts ===")
    _clean_output_directory(output_dir)

    for gen_name, generator in GENERATORS.items():
        gen_dir = output_dir / gen_name
        gen_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, prompts in loaded.items():
            _save_dataset_prompts(
                dataset_name=dataset_name,
            prompts=prompts,
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
        seed=settings.common.seed,
        output_dir=Path(settings.common.prompt_dir),
    )


if __name__ == "__main__":
    main()
