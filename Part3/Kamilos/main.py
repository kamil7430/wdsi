import json

import model

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
        result_struct = model.extract_gpu_criteria(q)

        print("Wyciągnięta struktura danych (AI Struct):")
        print(json.dumps(result_struct, indent=2, ensure_ascii=False))