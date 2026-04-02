import csv
import json

# Build dictionary of questions to ocena from CSV
ocena_dict = {}
with open('filtered-llmzszl - filtered-llmzszl.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ocena_dict[row['question']] = row['ocena']

# Filter JSONL based on ocena == '1'
with open('filtered-llmzszl.jsonl', 'r', encoding='utf-8') as f_in, \
     open('double-filtered-llmzszl.jsonl', 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line.strip())
        if data['question'] in ocena_dict and ocena_dict[data['question']] == '1':
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')