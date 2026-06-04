import model

# 1. BAZA WZORCOWA (Ground Truth) - Oczekiwane poprawne wyniki dla każdego zdania
EXPECTED_RESULTS = [
    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 12, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 16, "brand": "amd", "is_cheap": True},
    {"min_vram_gb": 8, "brand": None, "is_cheap": True},  # najtańsza = cheap
    {"min_vram_gb": None, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 24, "brand": "amd", "is_cheap": True},

    {"min_vram_gb": 24, "brand": None, "is_cheap": True},  # dwadzieścia cztery = 24
    {"min_vram_gb": 16, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 24, "brand": None, "is_cheap": True},
    {"min_vram_gb": 12, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 16, "brand": "amd", "is_cheap": True},  # szesnastoma/szestnastoma = 16

    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 12, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 16, "brand": "nvidia", "is_cheap": True},  # domyślnie bierzemy nvidia/pierwszą ze słownika przy OR
    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 8, "brand": "nvidia", "is_cheap": True},

    {"min_vram_gb": 12, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 16, "brand": "amd", "is_cheap": True},
    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 8, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 12, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},

    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 16, "brand": "amd", "is_cheap": True},
    {"min_vram_gb": 24, "brand": "nvidia", "is_cheap": True},
    {"min_vram_gb": 8, "brand": "nvidia", "is_cheap": False},  # droga = False
    {"min_vram_gb": 24, "brand": "amd", "is_cheap": False},  # cena nie gra roli = False
    {"min_vram_gb": 4, "brand": None, "is_cheap": True},
]

TEST_SENTENCES = [
    "Tanie GPU do trenowania modeli językowych, najlepiej z dużą ilością 24 GB vram i dobrym wsparciem dla cuda.",
    "Szukam taniej karty od nvidia z minimum 12gb pamięci.",
    "Potrzebuję budżetowe gpu do stabilnej dyfuzji, najlepiej amd 16 gb vram.",
    "Jaka jest najtańsza karta graficzna posiadająca przynajmniej 8gb vram?",
    "Interesuje mnie tania karta nvidia do głębokiego uczenia.",
    "Szukam czegoś od amd z 24gb pamięci w niskiej cenie.",
    "Chciałbym kupić niedrogie gpu, które ma przynajmniej dwadzieścia cztery gigabajty pamięci.",
    "Znajdź mi coś w ekonomicznej cenie ze stajni nvidia, co ma 16gb na pokładzie.",
    "Potrzebuję karty z dużą ilością vramu, najlepiej 24 gigabajtów, ale w niska cena.",
    "Interesują mnie wyłącznie tanich kart od nvidia, najlepiej z pamięcią minimum 12 gb.",
    "Jakaś ekonomiczna opcja z technologią cuda i dużym vram (24gb)?",
    "Poszukuję budżetowej architektury z szesnastoma gigabajtami pamięci od amd.",
    "CUDA jest dla mnie ważna, vram minimum 24gb, ale żeby cena za godzinę była niska i tania.",
    "Do LLM, 12gb vram wystarczy, nvidia, cena ma być jak najniższa.",
    "AMD lub nvidia, bez różnicy, byleby miało 16 gb i było tanie w eksploatacji.",
    "Pamięć: 24 gb. Budżet: tania. Producent: nvidia. Co polecasz?",
    "Szukam czegoś niedrogiego, zależy mi na technologii cuda oraz pamięci rzędu 24 gigabajtów.",
    "8 gb vram to absolutne minimum, marka nvidia, szukam czegoś w budżetowej półce.",
    "Cześć, jestem studentem i robię projekt na zajęcia z AI, potrzebuję tanie gpu z 12gb vram od nvidia, pomożesz?",
    "Do gier i pracy przy sieciach neuronowych szukam niedrogiej karty, najlepiej 16gb vram, może być amd.",
    "Moje fundusze są ograniczone, więc interesuje mnie wyłącznie niska cena i 24gb vram z obsługą cuda.",
    "Profesor kazał nam znaleźć jakieś tanie rozwiązanie z 8 gigabajtami pamięci od nvidia, co macie?",
    "Ekologiczne i przede wszystkim ekonomiczne gpu, nvidia z pamięcią 12 gb.",
    "Chcę postawić lokalnego bota, więc szukam taniego rozwiązania z 24gb vram, najlepiej nvidia.",
    "Potrzebuję gpu nvidia rtx 3090, niska cena, pamięć 24gb.",
    "Tanie amd 16gb.",
    "Nvidia cuda 24 gb vram tanio.",
    "Szukam drogiej karty nvidia, która ma mało vramu, np. 8 gb.",
    "Potrzebuję super wydajnego potwora od amd z 24gb vram, cena nie gra roli.",
    "Jakaś tania karta z 4gb vram dla mało wymagającego modelu.",
]


def run_evaluation_tests():
    total_tests = len(TEST_SENTENCES)
    passed_tests = 0

    print("=" * 70)
    print("                URUCHAMIANIE TESTÓW EWALUACYJNYCH                  ")
    print("=" * 70)

    for idx, (sentence, expected) in enumerate(zip(TEST_SENTENCES, EXPECTED_RESULTS), 1):
        actual = model.extract_gpu_criteria(sentence)

        is_correct = actual == expected

        if is_correct:
            passed_tests += 1
            status = "✅ OK"
        else:
            status = "❌ BŁĄD"

        print(f"[{idx:02d}] {status} | Prompt: \"{sentence[:50]}...\"")
        if not is_correct:
            print(f"     -> Oczekiwano: {expected}")
            print(f"     -> Otrzymano : {actual}")

    print("-" * 70)
    accuracy = (passed_tests / total_tests) * 100
    print(f"PODSUMOWANIE:")
    print(f"Poprawnie zinterpretowane: {passed_tests} z {total_tests}")
    print(f"Skuteczność modelu (Accuracy): {accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation_tests()