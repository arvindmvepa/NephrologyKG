import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def append_files_list(file_names):
    list = []
    for names in file_names:
        with open(names) as infile:
            file = open(names, encoding = 'utf8') 
            data = file.read() 
            file.close()
            list.append(data)
    return list

doc_complete = append_files_list(['C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renovascular_hypertension.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_neurotic_hypertension.txt', 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_antiglomerular_basement.txt'])



def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=150)

print(ldamodel.print_topics(num_topics=20, num_words=3))