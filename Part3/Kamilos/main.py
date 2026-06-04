import json
import re
import pl_core_news_sm

# 1. ŁADOWANIE LOKALNEGO MODELU DLA JĘZYKA POLSKIEGO
# Model ten potrafi już sam z siebie rozpoznawać podstawowe części mowy i formy podstawowe (lematy)
nlp = pl_core_news_sm.load()

# 2. DEFINIOWANIE REGUŁ NLP DLA PARAMETRÓW GPU (EntityRuler)
# Tworzymy dedykowany komponent, który nauczy model rozpoznawać nasze specyficzne encje
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    # Wzorce dla VRAM (np. "24 GB", "16gb", "24 gigabajty")
    {"label": "VRAM", "pattern": [{"TEXT": {"REGEX": r"^\d+$"}}, {"LOWER": {"IN": ["gb", "giga", "gigabajtów", "gigabajty", "vram"]}}]},
    {"label": "VRAM", "pattern": [{"TEXT": {"REGEX": r"^\d+(gb|giga|vram)$"}}]},

    # Wzorce dla producenta / technologii
    {"label": "BRAND", "pattern": [{"LOWER": "nvidia"}]},
    {"label": "BRAND", "pattern": [{"LOWER": "amd"}]},
    {"label": "BRAND", "pattern": [{"LOWER": "cuda"}]},  # skojarzymy cuda z nvidia

    # Wzorce dla budżetu
    {"label": "BUDGET", "pattern": [{"LOWER": {"IN": ["tanie", "tania", "tani", "budżetowe", "niedrogie", "ekonomiczne"]}}]},
    {"label": "BUDGET", "pattern": [{"LOWER": "niska"}, {"LOWER": "cena"}]}
]

ruler.add_patterns(patterns)


# 3. FUNKCJA EKSTRAKCJI CECH (Nasz "Parser AI")
def extract_gpu_criteria(user_prompt: str) -> dict:
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


# ==========================================
# TEST SYSTEMU NLP
# ==========================================
if __name__ == "__main__":
    test_queries = [
        "Tanie GPU do trenowania modeli językowych, najlepiej z dużą ilością 24 GB vram i dobrym wsparciem dla cuda",
        "Potrzebuję drogiej karty AMD która ma przynajmniej 16gb pamięci"
    ]

    for q in test_queries:
        print(f"\nWejściowy prompt: '{q}'")
        result_struct = extract_gpu_criteria(q)

        print("Wyciągnięta struktura danych (AI Struct):")
        print(json.dumps(result_struct, indent=2, ensure_ascii=False))