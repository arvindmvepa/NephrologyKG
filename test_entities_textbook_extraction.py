#import scispacy
import spacy

nlp = spacy.load("en_core_sci_sm")
text = """
Optimal management of patients with chronic kidney disease (CKD) requires appropriate 
interpretation and use of the markers and stages of CKD, early disease recognition, and 
collaboration between primary care physicians and nephrologists. Because multiple terms have 
been applied to chronic kidney disease (CKD), eg, chronic renal insufficiency, chronic renal 
disease, and chronic renal failure, the National Kidney Foundation Kidney Disease Outcomes 
Quality Initiative™ (NKF KDOQI™) has defined the all-encompassing term, CKD. Using kidney 
rather than renal improves understanding by patients, families, healthcare workers, and the lay 
public. This term includes the continuum of kidney dysfunction from mild kidney damage to 
kidney failure, and it also includes the term, end-stage renal disease (ESRD).
"""
doc = nlp(text)
print(list(doc.sents))

# Examine the entities extracted by the mention detector.
# Note that they don't have types like in SpaCy, and they
# are more general (e.g including verbs) - these are any
# spans which might be an entity in UMLS, a large
# biomedical database.
#print(doc.ents)
for ent in doc.ents:
    print(ent)

# We can also visualise dependency parses
# (This renders automatically inside a jupyter notebook!):
from spacy import displacy
displacy.render(next(doc.sents), style='dep', jupyter=True)