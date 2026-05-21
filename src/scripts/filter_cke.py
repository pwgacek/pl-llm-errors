#!/usr/bin/env python3
"""
Filter otwarte-pytania-matura-cke.jsonl using IDs from matury dataset.
"""

import json

# IDs to filter by
IDS = {'2025_maj_1', '2024_czerwiec_1', '2024_czerwiec_2', '2023_maj_1', '2016_maj_7', '2015_czerwiec_7', '2025_czerwiec_1', '2025_czerwiec_3', '2024_maj_3', '2024_maj_4', '2024_czerwiec_4', '2023_czerwiec_2', '2023_czerwiec_3', '2025_maj_8_stara_formula', '2024_maj_10_stara_formula', '2022_czerwiec_7', '2021_maj_7', '2020_lipiec_6', '2020_lipiec_13', '2019_czerwiec_6', '2016_czerwiec_10', '2025_maj_5', '2025_czerwiec_6', '2025_czerwiec_8', '2024_maj_9', '2024_maj_13_2', '2023_maj_8', '2023_maj_9', '2023_maj_10', '2023_czerwiec_8', '2023_czerwiec_9', '2024_maj_13_stara_formula', '2023_czerwiec_11_stara_formula', '2022_maj_10', '2022_czerwiec_10', '2022_czerwiec_11', '2021_czerwiec_11', '2020_czerwiec_11', '2018_maj_10', '2018_maj_11', '2018_czerwiec_9', '2017_maj_11', '2017_czerwiec_9', '2017_czerwiec_11', '2016_maj_11', '2016_czerwiec_15', '2015_maj_10', '2015_maj_11', '2015_czerwiec_10', '2015_czerwiec_11', '2025_maj_8', '2025_maj_9', '2025_czerwiec_9', '2024_maj_10', '2024_maj_11', '2023_czerwiec_11', '2022_maj_12', '2022_czerwiec_12', '2021_czerwiec_14', '2020_czerwiec_12', '2020_lipiec_11', '2018_czerwiec_12', '2018_czerwiec_13', '2017_maj_12', '2017_czerwiec_10', '2017_czerwiec_12', '2015_maj_13', '2015_maj_14', '2015_czerwiec_13', '2015_czerwiec_14', '2025_czerwiec_11', '2024_maj_12', '2024_czerwiec_11', '2024_czerwiec_12', '2023_maj_13', '2023_czerwiec_14_stara_formula', '2022_maj_14', '2022_czerwiec_14', '2021_maj_14', '2020_czerwiec_14', '2020_lipiec_12', '2019_maj_12', '2019_czerwiec_11', '2019_czerwiec_13', '2018_maj_12', '2018_czerwiec_14', '2017_maj_14', '2017_czerwiec_14', '2016_maj_15', '2015_czerwiec_15', '2021_maj_15', '2020_czerwiec_15', '2020_lipiec_15', '2019_czerwiec_15', '2018_czerwiec_15', '2017_maj_15', '2017_czerwiec_15', '2016_maj_16', '2015_maj_16', '2015_czerwiec_16'}

INPUT_FILE = "/home/pawel/Desktop/pl-llm-errors/datasets/otwarte-pytania-matura-cke.jsonl"
OUTPUT_FILE = "/home/pawel/Desktop/pl-llm-errors/datasets/otwarte-pytania-filtered.jsonl"


def main():
    count_total = 0
    count_matched = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            count_total += 1
            item = json.loads(line)
            
            if item.get('id') in IDS:
                count_matched += 1
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Filtered {count_matched}/{count_total} records")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
