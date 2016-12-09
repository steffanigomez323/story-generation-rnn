import numpy as np
import codecs
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re


doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def basic_tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
    """
    Very basic tokenizer: split the sentence into a list of tokens.
    """
    words = []
    space_fragments = re.findall(r'\S+|\n', sentence)
    for space_separated_fragment in space_fragments:
        # for space_separated_fragment in sentence.split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w for w in words if w]

# doc_clean = [clean(doc).split() for doc in doc_complete]
#
# # Creating the term dictionary of our courpus, where every unique term is assigned an index.
# dictionary = corpora.Dictionary(doc_clean)
#
# # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
# doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
#
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
#
# # Running and Training LDA model on the document term matrix.
# ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
#
# print(ldamodel.print_topics(num_topics=3, num_words=3))

initfile = 'data/drseuss.txt'

#doclen = np.random.poisson(lam=5, size=1)
doclen = 10

docs = []
with codecs.open(initfile, 'r', encoding='utf-8') as f:
    for line in f:
        #words = line.split()
        words = basic_tokenizer(line)
        for w in words:
            docs.append(w)
        #words = basic_tokenizer(line)
        #for w in words:
        #    text.append(w)
    #docs = f.read()

docs_complete = []

for i in range(0, len(docs), doclen):
    docs_complete.append(' '.join(docs[i:i+doclen]))

#print docs_complete

docs_complete_clean = [doc.split(" ") for doc in docs_complete]#[clean(doc).split() for doc in docs_complete]

#print docs_complete_clean

dictionary = corpora.Dictionary(docs_complete_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_complete_clean]

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=50, id2word = dictionary, passes=50)

#print(ldamodel.print_topics(num_topics=5, num_words=5))

print(ldamodel.show_topics(num_topics=5, num_words=3, formatted=False))

letstry = docs_complete_clean[0][:]
letstry.extend(docs_complete_clean[1])

print(letstry)
print(dictionary.doc2bow(letstry))
print(ldamodel.get_document_topics(dictionary.doc2bow(letstry)))

wholetext = []

for doc in docs_complete:
    for w in clean(doc).split():
        wholetext.append(w)

topics = ldamodel.get_document_topics(dictionary.doc2bow(wholetext), minimum_probability=0)


#print(topics)
print(len(topics))

def generatetext(model, num=50):
    res = []
    for i in range(num):
        topic = np.random.choice([int(i[0]) for i in topics], p=[float(i[1]) for i in topics])
        worddist = model.get_topic_terms(topic, len(dictionary.keys()))
        word = np.random.choice([int(i[0]) for i in worddist], p=[float(i[1]) for i in worddist])
        res.append(dictionary.get(word))
        #break
    print(' '.join(res))

generatetext(ldamodel)