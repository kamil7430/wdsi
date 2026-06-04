import json
import re
import pl_core_news_sm

# ładowanie lokalnego modelu dla języka polskiego
nlp = pl_core_news_sm.load()

# definiowanie reguł NLP dla parametrów GPU
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    # wzorzec dla vram
    {
        "label": "VRAM",
        "pattern": [
            {"TEXT": {"REGEX": r"^\d+$"}},
            {"LEMMA": {"IN": ["gb", "giga", "gigabajt", "vram", "ram"]}},
        ],
    },

    # wzorzec dla producenta/technologii
    {
        "label": "BRAND",
        "pattern": [
            {"LEMMA": {"IN": ["nvidia", "cuda", "amd", "intel"]}},
        ],
    },

    # wzorce dla budżetu
    {
        "label": "LOW_BUDGET",
        "pattern": [
            {"LEMMA": {"IN": ["tani", "budżetowy", "niedrogi", "ekonomiczny"]}},
        ],
    },
    {
        "label": "LOW_BUDGET",
        "pattern": [
            {"LEMMA": "niski"},
            {"LEMMA": "cena"},
        ],
    }
]

ruler.add_patterns(patterns)


def add_spaces_around_numbers(string: str) -> str:
    new_str = string[0]
    for i in range(1, len(string)):
        if string[i-1].isalpha() and string[i].isdigit():
            new_str += ' '
        if string[i-1].isdigit() and string[i].isalpha():
            new_str += ' '
        new_str += string[i]
    return new_str

# 3. FUNKCJA EKSTRAKCJI CECH (Nasz "Parser AI")
def extract_gpu_criteria(user_prompt: str) -> dict:
    user_prompt = add_spaces_around_numbers(user_prompt)

    doc = nlp(user_prompt)

    extracted = {
        "min_vram_gb": None,
        "brand": None,
        "is_cheap": False
    }

    # Przechodzimy przez rozpoznane przez spaCy encje (entities)
    for ent in doc.ents:
        if ent.label_ == "VRAM":
            # Wyciągamy samą liczbę z tekstu encji (np. z "24 GB" robi 24)
            digits = [int(s) for s in ent.text.split() if s.isdigit()]
            if not digits:  # obsługa przypiętego tekstu np. "24gb"
                match = re.search(r'\d+', ent.text)
                if match: digits = [int(match.group())]
            if digits:
                extracted["min_vram_gb"] = digits[0]

        elif ent.label_ == "BRAND":
            val = ent.text.lower()
            if val in ["nvidia", "cuda"]:
                extracted["brand"] = "nvidia"
            elif val == "amd":
                extracted["brand"] = "amd"

        elif ent.label_ == "BUDGET":
            extracted["is_cheap"] = True

    return extracted


if __name__ == "__main__":
    TEST_SENTENCES = [
        # Kategoria 1: Zapytania standardowe i bezpośrednie
        "Tanie GPU do trenowania modeli językowych, najlepiej z dużą ilością 24 GB vram i dobrym wsparciem dla cuda.",
        "Szukam taniej karty od nvidia z minimum 12gb pamięci.",
        "Potrzebuję budżetowe gpu do stabilnej dyfuzji, najlepiej amd 16 gb vram.",
        "Jaka jest najtańsza karta graficzna posiadająca przynajmniej 8gb vram?",
        "Interesuje mnie tania karta nvidia do głębokiego uczenia.",
        "Szukam czegoś od amd z 24gb pamięci w niskiej cenie.",

        # Kategoria 2: Trudniejsza gramatyka i synonimy
        "Chciałbym kupić niedrogie gpu, które ma przynajmniej dwadzieścia cztery gigabajty pamięci.",
        "Znajdź mi coś w ekonomicznej cenie ze stajni nvidia, co ma 16gb na pokładzie.",
        "Potrzebuję karty z dużą ilością vramu, najlepiej 24 gigabajtów, ale w niska cena.",
        "Interesują mnie wyłącznie tanich kart od nvidia, najlepiej z pamięcią minimum 12 gb.",
        "Jakaś ekonomiczna opcja z technologią cuda i dużym vram (24gb)?",
        "Poszukuję budżetowej architektury z szesnastoma gigabajtami pamięci od amd.",

        # Kategoria 3: Zapytania złożone i chaotyczne
        "CUDA jest dla mnie ważna, vram minimum 24gb, ale żeby cena za godzinę była niska i tania.",
        "Do LLM, 12gb vram wystarczy, nvidia, cena ma być jak najniższa.",
        "AMD lub nvidia, bez różnicy, byleby miało 16 gb i było tanie w eksploatacji.",
        "Pamięć: 24 gb. Budżet: tania. Producent: nvidia. Co polecasz?",
        "Szukam czegoś niedrogiego, zależy mi na technologii cuda oraz pamięci rzędu 24 gigabajtów.",
        "8 gb vram to absolutne minimum, marka nvidia, szukam czegoś w budżetowej półce.",

        # Kategoria 4: Zapytania z szumem informacyjnym
        "Cześć, jestem studentem i robię projekt na zajęcia z AI, potrzebuję tanie gpu z 12gb vram od nvidia, pomożesz?",
        "Do gier i pracy przy sieciach neuronowych szukam niedrogiej karty, najlepiej 16gb vram, może być amd.",
        "Moje fundusze są ograniczone, więc interesuje mnie wyłącznie niska cena i 24gb vram z obsługą cuda.",
        "Profesor kazał nam znaleźć jakieś tanie rozwiązanie z 8 gigabajtami pamięci od nvidia, co macie?",
        "Ekologiczne i przede wszystkim ekonomiczne gpu, nvidia z pamięcią 12 gb.",
        "Chcę postawić lokalnego bota, więc szukam taniego rozwiązania z 24gb vram, najlepiej nvidia.",

        # Kategoria 5: Przypadki brzegowe i "podchwytliwe"
        "Potrzebuję gpu nvidia rtx 3090, niska cena, pamięć 24gb.",
        "Tanie amd 16gb.",
        "Nvidia cuda 24 gb vram tanio.",
        "Szukam drogiej karty nvidia, która ma mało vramu, np. 8 gb.",
        "Potrzebuję super wydajnego potwora od amd z 24gb vram, cena nie gra roli.",
        "Jakaś tania karta z 4gb vram dla mało wymagającego modelu.",
    ]

    for q in TEST_SENTENCES:
        print(f"\nWejściowy prompt: '{q}'")
        result_struct = extract_gpu_criteria(q)

        print("Wyciągnięta struktura danych (AI Struct):")
        print(json.dumps(result_struct, indent=2, ensure_ascii=False))