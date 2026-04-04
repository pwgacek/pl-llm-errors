import argparse
import json
import re
from pathlib import Path
from typing import Any

NAME_TRANSLATION = {
# Dan -> Daniel
    "za Danem": "za Danielem",
    "poniżej Dana": "poniżej Daniela",
    "powyżej Dana": "powyżej Daniela",
    
    # Rob -> Robert
    "za Robem": "za Robertem",
    "poniżej Roba": "poniżej Roberta",
    "powyżej Roba": "powyżej Roberta",
    
    # Mya -> Maria
    "za Myą": "za Marią",
    "poniżej Myi": "poniżej Marii",
    "powyżej Myi": "powyżej Marii",
    "poniżej Mya": "poniżej Marii", 
    "powyżej Mya": "powyżej Marii", 
    
    # Amy -> Amelia 
    "za Amy": "za Amelią",
    "poniżej Amy": "poniżej Amelii",
    "powyżej Amy": "powyżej Amelii",
    
    # Joe -> Jan 
    "za Joe": "za Janem",
    "poniżej Joe": "poniżej Jana",
    "powyżej Joe": "powyżej Jana",

    # Eli -> Emil (DODANE ZABEZPIECZENIE)
    "za Elim": "za Emilem",
    "poniżej Eliego": "poniżej Emila",
    "powyżej Eliego": "powyżej Emila",

    # Mel -> Maciej (DODANE ZABEZPIECZENIE)
    "za Melem": "za Maciejem",
    "poniżej Mela": "poniżej Macieja",
    "powyżej Mela": "powyżej Macieja",

    # Ada -> Ada 
    "za Adą": "za Adą",
    "przed Adą": "przed Adą",
    "poniżej Ady": "poniżej Ady",
    "powyżej Ady": "powyżej Ady",

    # --- Na końcu formy podstawowe (Mianownik) ---
    "Ana": "Anna",
    "Eve": "Ewa",
    "Ada": "Ada",
    "Dan": "Daniel",
    "Rob": "Robert",
    "Amy": "Amelia",
    "Joe": "Jan",
    "Eli": "Emil",
    "Mel": "Maciej",
    "Mya": "Maria"
}


def _build_case_mapping(mapping: dict[str, str]) -> dict[str, str]:
    case_mapping = dict(mapping)
    for source, target in mapping.items():
        case_mapping[source.lower()] = target.lower()
    return case_mapping


def _replace_names_in_text(text: str, mapping: dict[str, str], pattern: re.Pattern[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        return mapping[match.group(0)]

    return pattern.sub(repl, text)


def _replace_names_in_obj(obj: Any, mapping: dict[str, str], pattern: re.Pattern[str]) -> Any:
    if isinstance(obj, dict):
        return {key: _replace_names_in_obj(value, mapping, pattern) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_names_in_obj(item, mapping, pattern) for item in obj]
    if isinstance(obj, str):
        return _replace_names_in_text(obj, mapping, pattern)
    return obj


def translate_jsonl(input_path: Path, output_path: Path) -> None:
    mapping = _build_case_mapping(NAME_TRANSLATION)
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(key) for key in keys) + r")\b")

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                dst.write("\n")
                continue

            record = json.loads(stripped)
            translated = _replace_names_in_obj(record, mapping, pattern)
            dst.write(json.dumps(translated, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate selected names in BBH JSONL file.")
    parser.add_argument(
        "--input",
        default="datasets/bbh.jsonl",
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="datasets/bbh3.jsonl",
        help="Path to output JSONL file.",
    )
    args = parser.parse_args()

    translate_jsonl(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
