import json

def find_duplicates_by_input(input_file, output_file):
    # Słownik do przechowywania już widzianych tekstów z pola 'input'
    # Klucz: string (zawartość input), Wartość: lista numerów linii
    seen_inputs = {}

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_number, line in enumerate(f_in, 1):  # Numerujemy linie od 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                text_input = data.get("input")

                if text_input is not None:
                    if text_input in seen_inputs:
                        # Znaleziono duplikat - wypisujemy do pliku
                        previous_lines = seen_inputs[text_input]
                        f_out.write(f"Linia {line_number} to duplikat dla pola 'input'. Wystąpił już w liniach: {previous_lines}\n")
                        
                        # Dodajemy obecną linię do historii
                        seen_inputs[text_input].append(line_number)
                    else:
                        # Pierwsze wystąpienie - dodajemy do słownika
                        seen_inputs[text_input] = [line_number]

            except json.JSONDecodeError:
                print(f"Ostrzeżenie: Nieprawidłowy format JSON w linii {line_number}")

if __name__ == "__main__":
    # Tutaj podaj nazwy swoich plików
    PLIK_WEJSCIOWY = "/home/pawel/Desktop/datasets/bbh-logical-deduction-seven-objects-pl/open.jsonl"
    PLIK_WYJSCIOWY = "duplicates_2.txt"
    
    find_duplicates_by_input(PLIK_WEJSCIOWY, PLIK_WYJSCIOWY)
    print("Gotowe! Wyniki zapisano w pliku", PLIK_WYJSCIOWY)