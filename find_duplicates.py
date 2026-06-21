import json

def find_duplicates_in_jsonl(input_file, output_file):
    # Słownik do przechowywania już widzianych kluczy 'correct_order'
    # Klucz: krotka (tuple) z wartościami, Wartość: lista numerów linii
    seen_orders = {}

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_number, line in enumerate(f_in, 1):  # Numerujemy linie od 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                correct_order = data.get("correct_order")

                if correct_order is not None:
                    # Zamieniamy listę list na krotkę krotek, by móc użyć jej jako klucza w słowniku
                    order_tuple = tuple(tuple(item) for item in correct_order)

                    if order_tuple in seen_orders:
                        # Znaleziono duplikat - wypisujemy do pliku
                        previous_lines = seen_orders[order_tuple]
                        f_out.write(f"Linia {line_number} to duplikat dla klucza 'correct_order'. Wystąpił już w liniach: {previous_lines}\n")
                        
                        # Dodajemy obecną linię do historii, aby kolejne duplikaty o niej "wiedziały"
                        seen_orders[order_tuple].append(line_number)
                    else:
                        # Pierwsze wystąpienie - dodajemy do słownika
                        seen_orders[order_tuple] = [line_number]

            except json.JSONDecodeError:
                print(f"Ostrzeżenie: Nieprawidłowy format JSON w linii {line_number}")

if __name__ == "__main__":
    # Tutaj podaj nazwy swoich plików
    PLIK_WEJSCIOWY = "/home/pawel/Desktop/datasets/bbh-logical-deduction-seven-objects-pl/open.jsonl"
    PLIK_WYJSCIOWY = "duplicates.txt"
    
    find_duplicates_in_jsonl(PLIK_WEJSCIOWY, PLIK_WYJSCIOWY)
    print("Gotowe! Wyniki zapisano w pliku", PLIK_WYJSCIOWY)