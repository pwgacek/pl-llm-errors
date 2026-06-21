import json

def add_id_to_jsonl(input_file, output_file):
    processed_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        # Funkcja enumerate(f_in, 0) domyślnie zaczyna liczyć od 0
        for current_id, line in enumerate(f_in, 0):
            line = line.strip()
            if not line:
                continue

            try:
                # Wczytanie oryginalnego rekordu
                record = json.loads(line)
                
                # Tworzymy nowy słownik, aby upewnić się, że 'id' będzie pierwszym polem
                new_record = {"id": current_id}
                
                # Doklejamy resztę starych danych z oryginalnego rekordu
                new_record.update(record)
                
                # Zapisujemy nowy rekord do pliku wyjściowego
                json.dump(new_record, f_out, ensure_ascii=False)
                f_out.write('\n')
                
                processed_count += 1

            except json.JSONDecodeError:
                print(f"Ostrzeżenie: Nieprawidłowy format JSON w linii {current_id + 1}")

    print(f"Zakończono! Dodano pole 'id' do {processed_count} rekordów.")
    print(f"Wynik zapisano w pliku: {output_file}")

if __name__ == "__main__":
    # Ścieżki do plików
    PLIK_WEJSCIOWY = "/home/pawel/Desktop/pl-llm-errors/open-filtered.jsonl"
    PLIK_WYJSCIOWY = "/home/pawel/Desktop/pl-llm-errors/open-filtered2.jsonl"
    
    add_id_to_jsonl(PLIK_WEJSCIOWY, PLIK_WYJSCIOWY)