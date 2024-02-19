import spacy
from scispacy.linking import EntityLinker
from spacy.tokens import Span

EntityLinker.name = 'scispacy_linker'
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
doc_string = "GDG\nevidence\nadverse outcomes\npeople\nCKD\nGFR\nCKD\nmanagement\nprognosis\npeople" + \
			 "\nreduced but stable\nGFR\nprogressive decline\nGFR\nevidence\ndecline\nGFR\nkidney disease\n" + \
			 "‘natural’ decline\nageing"
doc = nlp(doc_string)

# get linking results on mentions mined by LLM and then pre-processed by spacy methods
linker = nlp.get_pipe("scispacy_linker")
print("results from spacy using their NER and linking on mentions")
for entity in doc.ents:
	if entity._.kb_ents:
		cui = entity._.kb_ents[0][0]
		print(str(entity).strip() + ", " + str(cui))
	else:
		print(str(entity).strip() + ", None")


print("results from spacy using no NER and linking on mentions")
entities = doc_string.split("\n")
spans = [Span(doc, start, start+len(entity.split()), label="ENTITY") for start, entity in enumerate(entities)]
for span in spans:
	entity = linker(span)
	if span._.kb_ents:
		cui = entity._.kb_ents[0][0]
		print(str(span.text).strip() + ", " + cui)
	else:
		print(str(span.text).strip() + ", None")
