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
from stemmer import *
from dataset_generator import read_file
from dictionary_generator import *
from prediction import predict

# main function
def main():
	# Taking parameters from command line
	train_data_path = sys.argv[1]
	test_data_path = sys.argv[2]
	part = sys.argv[3]
  	
	if part=='a':
		# reading training and test data from .json file
  		(train_X,train_Y) = read_file(train_data_path)
  		(test_X,test_Y) = read_file(test_data_path)
  		
	  	# creating the dictionary i.e. for keeping count of words
		(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y,"split")
	  	
	  	# prediction time on test_X
	  	prediction = predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary,"split")
	  	
	  	test_Y_array = [0]*len(test_Y)
	  	for i in range(len(test_Y)):
	  		test_Y_array[i] = test_Y[i]

	  	confatrix = confusion_matrix(test_Y_array,prediction)
	  	f1_matrix = f1_score(test_Y_array,prediction,average=None)
	  	print("F1 Score")
	  	print(f1_matrix)
	  	print("Confusion Matrix")
	  	print(confatrix)
	  	macro_f1 = f1_score(test_Y_array,prediction,average='macro')
	  	print("Macro F1 Score")
	  	print(macro_f1)
	  	# draw_confusion(confatrix)

	elif part=='b':
		# means random and majority prediction
		# reading training and test data from .json file
  		(train_X,train_Y) = read_file(train_data_path)
  		(test_X,test_Y) = read_file(test_data_path)
  		
  		# creating the dictionary i.e. for keeping count of words
	  	(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y,"split")
	  	
	  	test_Y_array = [0]*len(test_Y)
	  	for i in range(len(test_Y)):
	  		test_Y_array[i] = test_Y[i]

	  	max_occuring_class = 1+np.argmax(class_occurences)
  		majority_prediction = [max_occuring_class] * len(test_X)
  		confatrix = confusion_matrix(test_Y_array,majority_prediction)
  		print("Confusion Matrix for majority prediction")
	  	print(confatrix)
	  	# draw_confusion(confatrix)

	  	random_prediction = np.random.random_integers(1,5,(len(test_X),1))
	  	confatrix = confusion_matrix(test_Y_array,random_prediction)
	  	print("Confusion Matrix for random prediction")
	  	print(confatrix)
	  	# draw_confusion(confatrix)

	elif part=='d':
		# reading training and test data from .json file
  		(train_X,train_Y) = read_file(train_data_path)
  		(test_X,test_Y) = read_file(test_data_path)
  		
  		# creating the dictionary i.e. for keeping count of words
	  	(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y,"stemming")
	  	
	  	# prediction time on test_X
	  	prediction = predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary,"stemming")
	  	
	  	test_Y_array = [0]*len(test_Y)
	  	for i in range(len(test_Y)):
	  		test_Y_array[i] = test_Y[i]

	  	confatrix = confusion_matrix(test_Y_array,prediction)
	  	f1_matrix = f1_score(test_Y_array,prediction,average=None)
	  	macro_f1 = f1_score(test_Y_array,prediction,average='macro')
	  	print("Confusion Matrix")
	  	print(confatrix)
	  	print("F1 Score")
	  	print(f1_matrix)
	  	print("Macro F1 Score")
	  	print(macro_f1)
	  	# draw_confusion(confatrix)

	elif part=='e':
		# feature_name = "stemming"
		feature_name = "split"
		
		# reading training and test data from .json file
  		(train_X,train_Y) = read_file(train_data_path)
  		(test_X,test_Y) = read_file(test_data_path)
  		
  		# creating the dictionary i.e. for keeping count of words
	  	(dictionary,class_occurences,class_vocabulary) = generate_dictionary(train_X,train_Y,feature_name)
	  	
	  	# prediction time on test_X
	  	prediction = predict(dictionary,test_X,test_Y,class_occurences,class_vocabulary,feature_name)
	  	
	  	test_Y_array = [0]*len(test_Y)
	  	for i in range(len(test_Y)):
	  		test_Y_array[i] = test_Y[i]

	  	confatrix = confusion_matrix(test_Y_array,prediction)
	  	print("Confusion Matrix")
	  	print(confatrix)
	  	f1_matrix = f1_score(test_Y_array,prediction,average=None)
	  	print("F1 Score")
	  	print(f1_matrix)
	  	macro_f1 = f1_score(test_Y_array,prediction,average='macro')
	  	print("Macro F1 Score")
	  	print(macro_f1)
	  	# draw_confusion(confatrix)

	else:
		print("No such part")

if __name__ == "__main__":
	main()


