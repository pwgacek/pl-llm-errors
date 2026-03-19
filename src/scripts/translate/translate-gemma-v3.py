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
        - NAMES GENDER: 
          * Female: Ana, Eve, Ada, Amy, Mya (zajęła, była).
          * Male: Dan, Rob, Joe, Eli, Mel (zajął, był).
        - RANKING: 'Third-to-last' -> trzeci od końca, 'Second-to-last' -> przedostatni.
        - VERBS: Use 'zajął/zajęła ... miejsce' or 'ukończył/ukończyła rywalizację'. Be consistent.""",
        "example": "Input: Ada finished second-to-last. Options: (A) Ada finished second-to-last\nOutput: Ada zajęła przedostatnie miejsce. Opcje: (A) Ada zajęła przedostatnie miejsce"
    },
    "BIRDS": {
        "keywords": ["birds", "branch", "hummingbird", "cardinal", "jay", "raven", "quail", "robin", "falcon", "crow", "hawk"],
        "rules": """
        - VOCABULARY: 
          * Hummingbird -> koliber (NOT kolibrier!)
          * Cardinal -> kardynał
          * Blue jay -> sójka
          * Raven -> kruk
          * Crow -> wrona
          * Quail -> przepiórka
          * Robin -> rudzik
          * Falcon -> sokół
          * Hawk -> jastrząb
          * Owl -> sowa
        - LOGIC: Translate numbers (second, third, fourth) EXACTLY. Do not change 2nd to 3rd.""",
        "example": "Input: The hawk is rightmost. Options: (A) The hawk is rightmost\nOutput: Jastrząb jest najbardziej po prawej. Opcje: (A) Jastrząb jest najbardziej po prawej"
    },
    "CARS": {
        "keywords": ["car show", "vehicles", "limousine", "sedan", "hatchback", "station wagon", "minivan", "convertible", "tractor"],
        "rules": """
        - VOCABULARY: 
          * Station wagon -> kombi
          * Convertible -> kabriolet
          * Sedan -> sedan
          * Hatchback -> hatchback
          * Limousine -> limuzyna
          * Tractor -> ciągnik
          * Minivan -> minivan
        - RANKING: 'Fourth-newest' -> czwarty najnowszy, 'Third-oldest' -> trzeci najstarszy.""",
        "example": "Input: The station wagon is fourth-newest. Options: (A) The station wagon is fourth-newest\nOutput: Kombi jest czwartym najnowszym pojazdem. Opcje: (A) Kombi jest czwartym najnowszym pojazdem"
    },
    "FRUIT": {
        "keywords": ["fruit", "stand", "mangoes", "loquats", "cantaloupes", "watermelons"],
        "rules": """
        - VOCABULARY: 
          * Loquat -> lokwata
          * Cantaloupe -> kantalupa 
          * Watermelon -> arbuz
          * Plum -> śliwka
          * Pear -> gruszka
          * Peach -> brzoskwinia
        - PRICES: 'Third-cheapest' -> trzeci najtańszy, 'Second-most expensive' -> drugi najdroższy.""",
        "example": "Input: Loquats are third-cheapest. Options: (A) Loquats are third-cheapest\nOutput: Lokwaty są trzecim najtańszym owocem. Opcje: (A) Lokwaty są trzecimi najtańszymi"
    },
    "BOOKS": {
        "keywords": ["shelf", "books", "color"],
        "rules": "Vocabulary: Brown -> brązowa, Gray -> szara, Purple -> fioletowa. 'Rightmost' -> najbardziej po prawej, 'Leftmost' -> najbardziej po lewej.",
        "example": "Input: The gray book is third. Options: (A) The gray book is third\nOutput: Szara książka jest trzecia. Opcje: (A) Szara książka jest trzecia"
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
    
    # Load existing results if file exists
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    
    for idx, row in df.tail(10).iterrows():
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
            
            result_item = {
                "id": idx,
                "category": category,
                "original": input_text,
                "translated": full_translated,
            }
            results.append(result_item)
            
            # Save after every text
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"[{category}] Processed row {idx} - Saved")
    
except Exception as e:
    print(f"Error: {e}")