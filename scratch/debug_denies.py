import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The patient denies any chest pain")
for token in doc:
    print(f"{token.text} ({token.dep_}, {token.pos_}) -> {token.head.text}")
    for child in token.children:
        print(f"  CHILD: {child.text} ({child.dep_})")
