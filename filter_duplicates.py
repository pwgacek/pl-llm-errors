import re

def filter_duplicates(log_file, input_jsonl, output_jsonl):
    # Zbiór do przechowywania numerów linii, które chcemy usunąć
    lines_to_remove = set()

    # Krok 1: Odczytanie numerów linii z pliku tekstowego z logami
    with open(log_file, 'r', encoding='utf-8') as f_log:
        for line in f_log:
            # Szukamy wzorca "Linia X to duplikat" na początku każdej linijki
            match = re.search(r"^Linia (\d+)", line.strip())
            if match:
                # Zamieniamy znaleziony tekst na liczbę całkowitą i dodajemy do zbioru
                line_num = int(match.group(1))
                lines_to_remove.add(line_num)

    # Krok 2: Przefiltrowanie pliku JSONL
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
         open(output_jsonl, 'w', encoding='utf-8') as f_out:

        for line_number, line in enumerate(f_in, 1):  # Numerujemy linie od 1
            # Zapisujemy linię tylko, jeśli jej numeru nie ma w zbiorze do usunięcia
            if line_number not in lines_to_remove:
                f_out.write(line)

    print(f"Zakończono! Usunięto {len(lines_to_remove)} duplikatów.")
    print(f"Oczyszczone dane zapisano w: {output_jsonl}")

if __name__ == "__main__":
    # Ścieżki do plików - zmodyfikuj je zgodnie z potrzebami
    PLIK_Z_LOGAMI = "duplicates.txt"
    PLIK_WEJSCIOWY_JSONL = "/home/pawel/Desktop/datasets/bbh-logical-deduction-seven-objects-pl/open.jsonl"
    PLIK_WYJSCIOWY_JSONL = "open-filtered.jsonl"
    
    filter_duplicates(PLIK_Z_LOGAMI, PLIK_WEJSCIOWY_JSONL, PLIK_WYJSCIOWY_JSONL)