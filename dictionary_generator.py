from nltk.stem.porter import PorterStemmer
from stemmer import *


# generating the dictionary
# Two ways either include the raw words or use the words after reducing them to stem
# Class occurences store the number of times each class has occured
# Class vocabulary store the number of words which occured in any document corresponding to that class
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