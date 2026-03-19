#!/usr/bin/env python3

import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

filename = "./test-00000-of-00001.parquet"
output_file = "bbh.jsonl"

try:
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B")
    
    # Read parquet file
    df = pd.read_parquet(filename)
    print("\nTranslating first row to Polish...")
    
    # Process only first row for testing
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.head(1).iterrows():
            input_text = row.get('input', '')
            
            # Translate to Polish
            inputs = tokenizer(input_text, return_tensors="pt")
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("pol_Latn"),
                max_length=256
            )
            translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            # Save to jsonl
            result = {
                "original": input_text,
                "translated": translated_text,
                "language": "Polish"
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Original: {input_text}")
            print(f"Translated: {translated_text}")
    
    print(f"\nResults saved to {output_file}")
    
except FileNotFoundError:
    print(f"Error: File '{filename}' not found")
except Exception as e:
    print(f"Error: {e}")
