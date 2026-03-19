#!/usr/bin/env python3

import pandas as pd
import json
import requests

filename = "src/scripts/translate/test-00000-of-00001.parquet"

try:
    # Read parquet file
    df = pd.read_parquet(filename)
    print("Translating documents...\n")
    
    # Collect results
    results = []
    
    # Process rows
    for idx, row in df.tail(70).iterrows():
            input_text = row.get('input', '')
            print(f"Row {idx}:\n{input_text}\n")
except Exception as e:
    print(f"Error: {e}")