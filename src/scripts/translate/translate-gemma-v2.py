#!/usr/bin/env python3

import pandas as pd
import json
import requests

# Config
filename = "src/scripts/translate/test-00000-of-00001.parquet"
output_file = "src/scripts/translate/bbh-gemma-12B_v3.json"
ollama_url = "http://localhost:11434/api/generate"
ollama_model = "translategemma:12b"

COMMON_PREFIX = "The following paragraphs each describe a set of seven objects arranged in a fixed order. The statements are logically consistent within each paragraph."
TRANSLATED_PREFIX = "Następne akapity opisują zestaw siedmiu obiektów ułożonych w ustalonej kolejności. Stwierdzenia są logicznie spójne w każdym akapicie."

# Category configurations with English instructions
CATEGORY_CONFIGS = {
    "GOLF": {
        "keywords": ["golf", "tournament", "golfers"],
        "rules": """
        - Grammar: Dan, Rob, Joe, Eli are male (use 'zajął', 'był'). Ana, Eve, Ada, Amy, Mya are female (use 'zajęła', 'była'). 
        - Logic: 'Third-to-last' is 'trzeci od końca'. NEVER use 'przedostatni' for this.
        - Logic: 'Second-to-last' is 'przedostatni'.""",
        "example": "Input: Rob finished third-to-last. Options: (A) Rob finished third-to-last\nOutput: Rob zajął trzecie miejsce od końca. Opcje: (A) Rob zajął trzecie miejsce od końca"
    },
    "BIRDS": {
        "keywords": ["birds", "branch"],
        "rules": """
        - Vocabulary: Raven -> kruk, Crow -> wrona, Owl -> sowa (write it as 'sowa', NOT 'sówa'!).
        - Precision: Translate numbers (second, third, fourth) EXACTLY as they are. 
        - Options: Each option (A-G) must feature the specific bird from the original text.""",
        "example": "Input: The owl is second. Options: (A) The owl is second\nOutput: Sowa jest druga. Opcje: (A) Sowa jest druga"
    },
    "CARS": {
        "keywords": ["car show", "vehicles", "newest", "oldest"],
        "rules": "Vocabulary: Station wagon -> kombi, Hatchback -> hatchback, Minivan -> minivan. Ranking: 'Fourth-newest' -> czwarty najnowszy, 'Third-oldest' -> trzeci najstarszy. Include 'Options:' section.",
        "example": "Input: The station wagon is the fourth-newest. Options: (A) The bus is the third-oldest\nOutput: Kombi jest czwartym najnowszym pojazdem. Opcje: (A) Autobus jest trzecim najstarszym."
    },
    "FRUIT": {
        "keywords": ["fruit stand", "fruits", "cheapest", "expensive"],
        "rules": "Vocabulary: Loquat -> lokwata, Plum -> śliwka. Prices: 'Third-cheapest' -> trzeci najtańszy, 'Second-most expensive' -> drugi najdroższy. Include 'Options:' section.",
        "example": "Input: The pears are the third-cheapest. Options: (A) The apples are the third-cheapest\nOutput: Gruszki są trzecim najtańszym owocem. Opcje: (A) Jabłka są trzecimi najtańszymi."
    },
    "BOOKS": {
        "keywords": ["shelf", "books", "leftmost", "rightmost"],
        "rules": "Vocabulary: Brown book -> brązowa książka. Positions: 'Second from the right' -> druga od prawej. Include 'Options:' section.",
        "example": "Input: The purple book is the rightmost. Options: (A) The brown book is the leftmost\nOutput: Fioletowa książka znajduje się najbardziej po prawej stronie. Opcje: (A) Brązowa książka jest najbardziej po lewej stronie."
    }
}

def detect_category(text):
    text_lower = text.lower()
    for category, config in CATEGORY_CONFIGS.items():
        if any(kw in text_lower for kw in config["keywords"]):
            return category
    return "DEFAULT"

try:
    df = pd.read_parquet(filename)
    results = []
    
    for idx, row in df.head(10).iterrows():
        input_text = row.get('input', '')
        clean_text = input_text.replace(COMMON_PREFIX, "").strip()
        
        category = detect_category(clean_text)
        config = CATEGORY_CONFIGS.get(category, {"rules": "Translate faithfully and naturally into Polish.", "example": ""})
        
        # Professional English prompt for Polish output
        prompt = f"""You are a professional translator from English to Polish.
Your task is to translate logic puzzles. Maintain perfect logical consistency.
You MUST translate the ENTIRE text, including the 'Options:' section.

Rules for this batch:
{config['rules']}

Follow this style:
{config['example']}

Translate the following text into Polish. Provide ONLY the translation:
{clean_text}"""
        
        response = requests.post(
            ollama_url,
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            translated_text = response.json()["response"].strip()
            full_translated = f"{TRANSLATED_PREFIX}\n\n{translated_text}"
            
            results.append({
                "category": category,
                "original": input_text,
                "translated": full_translated,
            })
            print(f"[{category}] Processed row {idx}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
except Exception as e:
    print(f"Error: {e}")