import spacy
from scispacy.linking import EntityLinker

EntityLinker.name = 'scispacy_linker'
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily.")

# Let's look at a random entity!
entity = doc.ents[1]

print("Name: ", entity)

# Each entity is linked to UMLS with a score
# (currently just char-3gram matching).
linker = nlp.get_pipe("scispacy_linker")
print("UMLS concepts: ", entity._.kb_ents)
for umls_ent in entity._.kb_ents:
	print(linker.kb.cui_to_entity[umls_ent[0]])