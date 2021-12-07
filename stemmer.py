from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import bigrams



def _stem(doc, p_stemmer, en_stop, return_tokens,feature_name):
    if feature_name=="bigram":
        tokens = word_tokenize(doc.lower())
        stemmed_tokens = bigrams(tokens)
    elif feature_name=="lemma":
        lemmatizer = WordNetLemmatizer()
        stemmed_tokens = lemmatizer.lemmatize(doc.lower())
    else:
        tokens = word_tokenize(doc.lower())
        stopped_tokens = filter(lambda token: token not in en_stop, tokens)
        stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
    if not return_tokens:
        return ' '.join(stemmed_tokens)
    return list(stemmed_tokens)

def getStemmedDocuments(docs,feature_name,return_tokens=True):
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    if isinstance(docs, list):
        output_docs = []
        for item in docs:
            output_docs.append(_stem(item, p_stemmer, en_stop, return_tokens,feature_name))
        return output_docs
    else:
        return _stem(docs, p_stemmer, en_stop, return_tokens,feature_name)