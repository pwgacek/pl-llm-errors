import json

input_file = 'datasets/llmzszl.jsonl'
output_file = 'src/scripts/open_llmzszl/filtered-llmzszl.jsonl'

filtered_data_list = []

with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        data = json.loads(line.strip())
        if data.get('type') in ['Egzaminy Gimnazjalne', 'Egzaminy Maturalne']:
            filtered_data = {
                'name': data.get('name'),
                'type': data.get('type'),
                'question': data.get('question'),
                'answers': data.get('answers'),
                'correct_answer_index': data.get('correct_answer_index'),
            }
            filtered_data_list.append(filtered_data)

# Sort by type, then by name
filtered_data_list.sort(key=lambda x: (x['type'], x['name']))

# Add id
for i, item in enumerate(filtered_data_list):
    item['id'] = i

# Reorder to have id first
for i in range(len(filtered_data_list)):
    item = filtered_data_list[i]
    filtered_data_list[i] = {'id': item['id'], **item}

# Write to output
with open(output_file, 'w', encoding='utf-8') as outfile:
    for item in filtered_data_list:
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')