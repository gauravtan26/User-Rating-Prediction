import re
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import bigrams
from nltk.stem import WordNetLemmatizer

# Necessary functionas copied from utils.py
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

def generate_dictionary(train_X,train_Y,feature_name):
	dictionary = {}
	num_words = len(train_X)
	class_occurences = [0,0,0,0,0]
	class_vocabulary = [0,0,0,0,0]
	if feature_name=="split":
		for i in range(len(train_X)):
			num_stars = train_Y[i]
			splitted_string = train_X[i].split()
			class_vocabulary[num_stars-1]+=len(splitted_string)
			class_occurences[num_stars-1]+=1
			for word in splitted_string:
				if word in dictionary:
					dictionary[word][num_stars-1]+=1
				else:
					dictionary[word] = [1,1,1,1,1]
					dictionary[word][num_stars-1]+=1
	else:
		for i in range(len(train_X)):
			num_stars = train_Y[i]
			splitted_string = getStemmedDocuments(train_X[i],feature_name,True)
			class_vocabulary[num_stars-1]+=len(splitted_string)
			class_occurences[num_stars-1]+=1
			for word in splitted_string:
				if word in dictionary:
					dictionary[word][num_stars-1]+=1
				else:
					dictionary[word] = [1,1,1,1,1]
					dictionary[word][num_stars-1]+=1

	return (dictionary,class_occurences,class_vocabulary)
