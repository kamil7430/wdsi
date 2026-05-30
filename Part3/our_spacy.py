import spacy
from spacy.symbols import nsubj, VERB


nlp = spacy.load("en_core_web_sm")

query = "I want not greater than 64 gb vram. price should be in range 200 to 500 cents per hour, from nvidia."
doc = nlp(query)
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])

print('---')
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

print('---')
for token in doc:
    print(token.text)

print('---')

# d = [token for token in doc if token.morph ]

print([token.lemma_ for token in doc])

USELESS = ["PRON", "VERB", "AUX", "ADP"]

new_sentence = " ".join(token.lemma_ for token in doc if token.pos_ not in USELESS)
print(new_sentence)

doc = nlp(new_sentence)
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])
