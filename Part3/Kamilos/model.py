import pl_core_news_lg

# ładowanie lokalnego modelu dla języka polskiego
nlp = pl_core_news_lg.load()

# dodanie reguł naprawczych do źle interpretowanych słów
attribute_ruler = nlp.get_pipe("attribute_ruler")
correct_lemmas = [
    {"patterns": [[{"LOWER": "gb"}]], "attrs": {"LEMMA": "gb"}},
    {"patterns": [[{"LOWER": "mb"}]], "attrs": {"LEMMA": "mb"}},
    {"patterns": [[{"LOWER": "cuda"}]], "attrs": {"LEMMA": "cuda"}},
]
for rule in correct_lemmas:
    attribute_ruler.add(patterns=rule["patterns"], attrs=rule["attrs"])

# definiowanie reguł NLP dla parametrów GPU
entity_ruler = nlp.add_pipe("entity_ruler", before="ner")
entity_ruler.add_patterns([
    # wzorzec dla vram
    {
        "label": "VRAM",
        "pattern": [
            {"TEXT": {"REGEX": r"^\d+$"}},
            {"LOWER": {"IN": ["mega", "megabajt", "megabajty", "megabajtów", "giga", "gigabajt",
                              "gigabajty", "gigabajtów", "mb", "gb", "ram", "vram", "megabajta",
                              "gigabajta"]}},
        ],
    },

    # wzorzec dla producenta/technologii
    {
        "label": "BRAND",
        "pattern": [
            {"LOWER": {"IN": ["nvidia", "nvidii", "nvidię", "cuda", "cudę", "cudą", "amd", "intel"]}},
        ],
    },

    # wzorce dla budżetu
    {
        "label": "LOW_BUDGET",
        "pattern": [
            {"LEMMA": {"IN": ["tać", "tani", "budżetowy", "niedrogi", "ekonomiczny"]}},
        ],
    },
    {
        "label": "LOW_BUDGET",
        "pattern": [
            {"LEMMA": "niski"},
            {"LEMMA": "cena"},
        ],
    }
])


def add_spaces_around_numbers(string: str) -> str:
    new_str = string[0]
    for i in range(1, len(string)):
        if string[i-1].isalpha() and string[i].isdigit():
            new_str += ' '
        if string[i-1].isdigit() and string[i].isalpha():
            new_str += ' '
        new_str += string[i]
    return new_str

def extract_gpu_criteria(user_prompt: str) -> dict:
    # preprocessing inputu
    user_prompt = add_spaces_around_numbers(user_prompt)
    user_prompt = user_prompt.lower()

    # przepuszczenie inputu użytkownika przez NLP spaCy
    doc = nlp(user_prompt)

    extracted = {
        "min_vram_gb": None,
        "brand": None,
        "is_cheap": False
    }

    # przejrzenie rozpoznanych przez spaCy encji
    for ent in doc.ents:
        if ent.label_ == "VRAM":
            digits = [int(s) for s in ent.text.split() if s.isdigit()]
            if digits:
                if digits[0] >= 1000:
                    extracted["min_vram_gb"] = digits[0] // 1000
                else:
                    extracted["min_vram_gb"] = digits[0]

        elif ent.label_ == "BRAND":
            val = ent.text.lower()
            if val in ["nvidia", "cuda"]:
                extracted["brand"] = "nvidia"
            elif val == "amd":
                extracted["brand"] = "amd"
            elif val == "intel":
                extracted["brand"] = "intel"

        elif ent.label_ == "LOW_BUDGET":
            extracted["is_cheap"] = True

    return extracted