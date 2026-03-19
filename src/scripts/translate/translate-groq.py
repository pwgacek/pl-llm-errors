#!/usr/bin/env python3

import pandas as pd
import json
import requests
import os

filename = "./test-00000-of-00001.parquet"
output_file = "bbh-qroq.json"
groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
groq_model = "llama-3.3-70b-versatile"
groq_api_key = os.environ["GROQ_API_KEY"]

# Translation settings
SOURCE_LANG = "English"
SOURCE_CODE = "en"
TARGET_LANG = "Polish"
TARGET_CODE = "pl"

# Common prefix in dataset
COMMON_PREFIX = "The following paragraphs each describe a set of seven objects arranged in a fixed order. The statements are logically consistent within each paragraph."
TRANSLATED_PREFIX = "Następne akapity opisują zestaw siedmiu obiektów ułożonych w ustalonej kolejności. Stwierdzenia są logicznie spójne w każdym akapicie."

try:
    # Read parquet file
    df = pd.read_parquet(filename)
    print("Translating documents...\n")
    
    # Collect results
    results = []
    
    # Process rows
    for idx, row in df.head(10).iterrows():
            input_text = row.get('input', '')
            
            # Remove common prefix from text
            if input_text.startswith(COMMON_PREFIX):
                clean_text = input_text[len(COMMON_PREFIX):].strip()
            else:
                clean_text = input_text
            
            # Create professional translation prompt for the meaningful content
            prompt = f"""You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:

{clean_text}"""
            
            # Call Groq API
            response = requests.post(
                groq_api_url,
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0  # Lower temperature for more deterministic translation
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Groq API request failed: {response.text}")
            
            translated_text = response.json()["choices"][0]["message"]["content"].strip()
            
            # Combine Polish prefix with translated text
            full_translated = TRANSLATED_PREFIX + "\n\n" + translated_text
            
            # Collect result
            result = {
                "original": input_text,
                "translated": full_translated,
            }
            results.append(result)
            print(f"Original (cleaned): {clean_text[:100]}...")
            print(f"Translated: {translated_text[:100]}...\n")
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
except FileNotFoundError:
    print(f"Error: File '{filename}' not found")
except requests.exceptions.ConnectionError:
    print(f"Error: Cannot connect to Groq API")
    print("Check your internet connection and API key")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure to set your Groq API key in the script")
