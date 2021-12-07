import time
import math
import numpy as np
from stemmer import *

# prediction function
def predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary,feature_name):
	num_test_points = len(test_X)
	num_train_points = sum(class_occurences)
	num_distinct_words = len(dictionary)
	prediction = [0] * num_test_points
	class_probabilities = [0.0]*5
	for j in range(5):
		class_probabilities[j]=math.log((float(class_occurences[j]))/(float(num_train_points)))

	for word in dictionary:
		for j in range(5):
			dictionary[word][j] = math.log((float(dictionary[word][j]))/(float(num_distinct_words+class_vocabulary[j])))

	if feature_name=="split":
		for i in range(num_test_points):
			prob = [0.0,0.0,0.0,0.0,0.0]
			for j in range(5):
				prob[j]+=class_probabilities[j]
				splitted_string = test_X[i].split()
				for word in splitted_string:
					if word in dictionary:
						prob[j]+=dictionary[word][j]
					else:
						prob[j]-=math.log((float(num_distinct_words+class_vocabulary[j])))
			prediction[i] = 1+np.argmax(prob)
	else:
		for i in range(num_test_points):
			prob = [0.0,0.0,0.0,0.0,0.0]
			for j in range(5):
				prob[j]+=class_probabilities[j]
				splitted_string = getStemmedDocuments(test_X[i],feature_name,True)
				for word in splitted_string:
					if word in dictionary:
						prob[j]+=dictionary[word][j]
					else:
						prob[j]-=math.log((float(num_distinct_words+class_vocabulary[j])))
			prediction[i] = 1+np.argmax(prob)
	return prediction
