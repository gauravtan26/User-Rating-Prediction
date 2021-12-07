import re
import json

# reading and cleansing file specified by the given path
def read_file(file_path):
	train_X = {}
	train_Y = {}
	counter = 0

	for line in open(file_path, mode="r"):
		line_contents = json.loads(line)
		review_text = line_contents["text"].strip().lower()
		review_text = re.sub(r'[^\w\s]','',review_text)
		review_text = re.sub(r'[^\w\s]','',review_text)
		review_text = re.sub('\r?\n',' ',review_text)
		train_X[counter] = review_text
		train_Y[counter] = int(line_contents["stars"])
		counter+=1
	return (train_X,train_Y)