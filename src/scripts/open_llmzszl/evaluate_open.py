import json
import os
import re
import time
from openai import OpenAI

# ---------------- KONFIGURACJA ----------------
client = OpenAI(
    base_url="https://api.groq.com/openai/v1", 
    api_key=os.getenv("GROQ_API_KEY")
)

input_file = 'src/scripts/open_llmzszl/double-filtered-llmzszl.jsonl'
output_file = 'src/scripts/open_llmzszl/evaluated-llmzszl.jsonl'

# ---------------- PROMPT SYSTEMOWY ----------------
system_prompt = """Jesteś edytorem testów matematycznych. 
Otrzymasz na wejściu JSON z oryginalnym pytaniem ("original_question") i opcjami ("options").
Twoim zadaniem jest zamiana go na naturalne, poprawne gramatycznie pytanie otwarte w języku polskim.

ZASADY ABSOLUTNE:
1. USUNĄĆ z tekstu frazy typu: "Dokończ zdanie tak, aby otrzymać zdanie prawdziwe.", "Dokończ zdanie.", "Wybierz właściwą odpowiedź spośród podanych."
2. NIE DODAWAĆ słowa "Pytanie: " na początku.
3. Jeśli pytanie kończy się zwrotem "...jest równa" lub "...wynosi", przeredaguj to zdanie na naturalne pytanie typu "Ile wynosi..." lub "Jaką wartość ma...".
4. Zachowaj wszystkie dane liczbowe, wzory oraz wstęp do zadania bez zmian.

PRZYKŁADY WEJŚCIA I WYJŚCIA:

Wejście: {"original_question": "Dokończ zdanie tak, aby otrzymać zdanie prawdziwe. Liczba (3^2 + 3^2 + 3^2)/3^3 jest równa", "options": ["1", "2", "3"]}
Wyjście: {"opened_question": "Ile wynosi wartość wyrażenia (3^2 + 3^2 + 3^2)/3^3?"}

Wejście: {"original_question": "Średnia odległość Marsa od Słońca wynosi 2,28⋅ 10^8 km. Odległość ta zapisana bez użycia potęgi jest równa", "options": ["228 000 km", "228 000 000 km"]}
Wyjście: {"opened_question": "Średnia odległość Marsa od Słońca wynosi 2,28⋅ 10^8 km. Jak zapisać tę odległość bez użycia potęgi?"}

Wejście: {"original_question": "Dokończ zdanie tak, aby otrzymać zdanie prawdziwe. Odległość na osi liczbowej między największą i najmniejszą spośród liczb: 0, 3/4, -5/2, -2 jest równa", "options": ["1", "3 1/4", "5"]}
Wyjście: {"opened_question": "Ile wynosi odległość na osi liczbowej między największą i najmniejszą spośród liczb: 0, 3/4, -5/2, -2?"}

Zwróć TYLKO czysty wynikowy obiekt JSON:
{"opened_question": "treść"}"""

# ---------------- FUNKCJE POMOCNICZE ----------------
def extract_json_from_response(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

# ---------------- GŁÓWNA PĘTLA ----------------
processed_ids = set()

if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    existing_data = json.loads(line.strip())
                    processed_ids.add(existing_data.get('id'))
                except json.JSONDecodeError:
                    continue

print(f"Znaleziono {len(processed_ids)} już przetworzonych pytań. Zaczynamy...")

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
    for line in infile:
        if not line.strip():
            continue
            
        data = json.loads(line.strip())
        id_ = data.get('id')
        
        if id_ in processed_ids:
            continue

        question = data.get('question', '')
        answers_list = data.get('answers', [])
        
        # Wyciąganie poprawnej odpowiedzi jako tekstu
        correct_index = data.get('correct_answer_index')
        correct_answer_text = None
        if correct_index is not None and isinstance(correct_index, int) and 0 <= correct_index < len(answers_list):
            correct_answer_text = answers_list[correct_index]
        
        # Pakujemy dane w czysty JSON dla modelu
        input_payload = {
            "original_question": question,
            "options": answers_list
        }
        full_question_context = json.dumps(input_payload, ensure_ascii=False)

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_question_context}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            clean_json = extract_json_from_response(result_text)
            result = json.loads(clean_json)
            
            opened_question = result.get('opened_question')
            print(f"Sukces ID {id_}")

        except Exception as e:
            print(f"Błąd dla ID {id_}: {e}")
            opened_question = ""

        # --- ZAPISUJEMY ORYGINALNE METADANE ---
        output_data = {
            'id': id_,
            'name': data.get('name'),
            'type': data.get('type'),
            'original_question': question,
            'opened_question': opened_question,
            'correct_answer': correct_answer_text
        }
        
        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        outfile.flush()

        time.sleep(0.5)

print("\nPrzetwarzanie zakończone.")