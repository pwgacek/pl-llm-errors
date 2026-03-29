import json
import re
import itertools

def clean_name(name):
    # Usuwa przedimki 'a', 'an', 'the' tylko z początku nazwy
    return re.sub(r'^(a|an|the)\s+', '', name.lower()).strip()

# Mapowanie wszystkich pozycji z bazy BBH na indeksy (0 = najlepiej/najtaniej, 6 = najgorzej/najdrożej)
exact_positions = {
    "first": 0, "leftmost": 0, "cheapest": 0, "newest": 0,
    "second from the left": 1, "second-cheapest": 1, "second cheapest": 1, "second-newest": 1, "second newest": 1, "second": 1, 
    "third from the left": 2, "third-cheapest": 2, "third cheapest": 2, "third-newest": 2, "third newest": 2, "third": 2, 
    "fourth from the left": 3, "fourth from the right": 3, "fourth-cheapest": 3, "fourth cheapest": 3, "fourth-newest": 3, "fourth newest": 3, "fourth-most expensive": 3, "fourth most expensive": 3, "fourth-oldest": 3, "fourth oldest": 3, "fourth-to-last": 3, "fourth to last": 3, "fourth": 3, 
    "third from the right": 4, "third-most expensive": 4, "third most expensive": 4, "third-oldest": 4, "third oldest": 4, "third-to-last": 4, "third to last": 4, "fifth": 4, 
    "second from the right": 5, "second-most expensive": 5, "second most expensive": 5, "second-oldest": 5, "second oldest": 5, "second-to-last": 5, "second to last": 5, "sixth": 5, 
    "rightmost": 6, "most expensive": 6, "oldest": 6, "last": 6
}

# Relacje między dwoma obiektami
rel_less = ["left of", "above", "newer than", "less expensive than"]
rel_more = ["right of", "below", "older than", "more expensive than"]

def solve_seven_objects(text):
    # 1. ROZDZIELAMY zadanie od opcji (żeby nie uczyć solvera z fałszywych wariantów A-G)
    parts = text.split("Options:")
    premises_text = parts[0]
    options_text = parts[1] if len(parts) > 1 else ""

    # 2. Wyciąganie obiektów (szukamy po dwukropku w pierwszym zdaniu)
    intro_match = re.search(r":\s*(.*?)\.", premises_text)
    if not intro_match: 
        return None, None, None
    
    raw_entities = [e.replace("and ", "").strip() for e in re.split(r",", intro_match.group(1))]
    raw_entities = [e for e in raw_entities if e]
    
    entities = [clean_name(e) for e in raw_entities]
    name_map = {clean_name(e): e for e in raw_entities} # Do odzyskania pełnej nazwy na końcu
    
    constraints = []
    # Rozbijamy warunki na zdania
    sentences = [s.strip().lower() for s in premises_text.split('.') if s.strip()]
    
    # 3. Parsowanie warunków
    for sentence in sentences[1:]: # Pomijamy zdanie nr 0, bo to tylko lista obiektów
        found_entities = []
        for e in entities:
            # Szukamy całych słów (żeby 'bus' nie łapało części innego słowa)
            if re.search(rf"\b{re.escape(e)}\b", sentence):
                found_entities.append(e)
        
        # Kluczowe! Sortujemy encje po kolejności ich wystąpienia W TYM ZDANIU
        found_entities.sort(key=lambda x: sentence.find(x))
        
        if len(found_entities) == 1:
            e1 = found_entities[0]
            # Szukamy najdłuższych dopasowań (żeby 'second-to-last' złapało przed 'last')
            for phrase, idx in sorted(exact_positions.items(), key=lambda x: -len(x[0])):
                if phrase in sentence:
                    constraints.append(lambda p, ent=e1, i=idx: p[ent] == i)
                    break
                    
        elif len(found_entities) == 2:
            e1, e2 = found_entities # e1 jest zawsze wymienione pierwsze w tekście
            matched = False
            for phrase in rel_less:
                if phrase in sentence:
                    constraints.append(lambda p, a=e1, b=e2: p[a] < p[b])
                    matched = True
                    break
            if not matched:
                for phrase in rel_more:
                    if phrase in sentence:
                        constraints.append(lambda p, a=e1, b=e2: p[a] > p[b])
                        break

    # 4. Obliczanie permutacji
    for perm in itertools.permutations(entities):
        p_map = {name: i for i, name in enumerate(perm)}
        if all(c(p_map) for c in constraints):
            return [name_map[name] for name in perm], entities, p_map
    
    return None, entities, None

def find_correct_option(p_map, entities, options_text):
    """Odczytuje, która odpowiedź A-G zgadza się z ustalonym rozwiązaniem."""
    if not p_map: return "Could not solve"
    
    options = re.findall(r"\(([A-G])\)\s*(.*)", options_text)
    for letter, opt_text in options:
        opt_text_lower = opt_text.lower()
        
        found_e = None
        for e in entities:
            if re.search(rf"\b{re.escape(e)}\b", opt_text_lower):
                found_e = e
                break
        
        if found_e:
            for phrase, idx in sorted(exact_positions.items(), key=lambda x: -len(x[0])):
                if phrase in opt_text_lower:
                    if p_map[found_e] == idx:
                        return f"({letter})"
                    break
    return "Solution found, but no matching option"

# =========================
# TEST NA TWOICH DANYCH
# =========================

def process_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    solved_data = []
    solved_count = 0

    for item in data:
        problem_text = item.get("input", "")
        item_id = item.get("id")

        order, entities, p_map = solve_seven_objects(problem_text)
        
        options_part = problem_text.split("Options:")[1] if "Options:" in problem_text else ""
        final_answer = find_correct_option(p_map, entities, options_part)

        if final_answer != "Could not solve":
            solved_count += 1
            
        solved_data.append({
            "id": item_id,
            "input": problem_text,
            "solution_order": order if order else "Could not solve",
            "final_answer": final_answer
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(solved_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nUdało się rozwiązać: {solved_count} / {len(data)}")
    print(f"Wyniki zapisano do pliku: {output_file}")

# Odkomentuj poniższą linijkę, aby przetworzyć swoje pliki
if __name__ == "__main__":
    process_dataset('./src/scripts/translate/bbh-english.json', './src/scripts/translate/bbh-english-solved.json')