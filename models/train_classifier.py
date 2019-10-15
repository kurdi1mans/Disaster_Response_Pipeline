import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')
nltk.download('omw')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet,stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,make_scorer,classification_report


import re
import pickle

def load_database(database_filepath):
	# load data from database
	engine = create_engine('sqlite:///'+database_filepath)
	df = pd.read_sql('SELECT * FROM message', engine)
	return df


def replace_URLs_with_placeholder(text):
	# Regular Expression to detect URLs for http and https urls (does not cater for uppercase HTTP/S or other protocols)
	url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
	
	#detect all URLs in a text message
	detected_urls = re.findall(url_regex, text)

	#replace URLs with 'urlplaceholder' string
	for url in detected_urls:
		text = text.replace(url, "urlplaceholder")
	return text

def tokenize_sentences_by_words(text):
	# this will make every sentence a token by itself
	sentence_list = nltk.sent_tokenize(text)
	
	# iterate through the sentences and make each one an array of token seperately.
	array_of_tokenized_sentences = []
	for sentence in sentence_list:
		word_tokenized_sentence = word_tokenize(sentence.lower())
		array_of_tokenized_sentences.append(word_tokenized_sentence)
	return array_of_tokenized_sentences

def tag_POS_for_sentence_tokens(array_of_tokenized_sentences):
	# take the array of tokens for each sentence seperately and get its POS tags
	array_of_tagged_sentence_tokens = []
	for sentence_tokens in array_of_tokenized_sentences:
		pos_tags = nltk.pos_tag(sentence_tokens)
		array_of_tagged_sentence_tokens.append(pos_tags)	
	return array_of_tagged_sentence_tokens

def lemmatize_tokens_based_on_POS_tags(array_of_tagged_sentence_tokens):
	# this mapping is from the POS tags to the wordnet tags understood by the lemmatization function
	tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}

	lemmatizer = WordNetLemmatizer()
	lemmatized_tokens = []
	for sentence_tokens in array_of_tagged_sentence_tokens:
		for token_pair in sentence_tokens:
			token = token_pair[0]
			stop_words = set(stopwords.words('english'))
			if (token not in stop_words) & token.isalpha():
				oldTag = token_pair[1].upper()
				newTag = tag_dict.get(oldTag, wordnet.NOUN)
				# Here we lemmatize based on the POS tag for better accuracy of lemmatization
				newToken = lemmatizer.lemmatize(token,newTag)
				lemmatized_tokens.append(newToken)
	return lemmatized_tokens

def tokenize(text):
	text = replace_URLs_with_placeholder(text)
	array_of_tokenized_sentences = tokenize_sentences_by_words(text)
	array_of_tagged_sentence_tokens = tag_POS_for_sentence_tokens(array_of_tokenized_sentences)
	lemmatized_tokens = lemmatize_tokens_based_on_POS_tags(array_of_tagged_sentence_tokens)
	return lemmatized_tokens

def train_valid_test_split(X,y):
	X_others, X_test, y_others, y_test = train_test_split(X, y,test_size=0.1, random_state = 42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_others, y_others,test_size=0.05, random_state = 42)
	return X_train,X_valid,X_test,y_train,y_valid,y_test


def get_train_valid_test_data(df):
	X = df['message']
	y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
	X_train,X_valid,X_test,y_train,y_valid,y_test = train_valid_test_split(X,y)
	return X_train,X_valid,X_test,y_train,y_valid,y_test

def print_model_metrics(y_pred,y_target,categories):
	y_target = pd.DataFrame(y_target,columns=categories)
	y_pred = pd.DataFrame(y_pred,columns=categories)
	
	for category in categories:
		print("Scores for Category '"+category+"'")
		temp = classification_report(y_target[category],y_pred[category])
		print(temp)


def create_model():
	pipeline = Pipeline(
							[
								('text_pipeline', Pipeline(
																[
																	('vect', CountVectorizer(tokenizer=tokenize)),
																	('tfidf', TfidfTransformer())

																]
															)),
								('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10,n_jobs=12)))
	
							]
						)
	RandomForest_parameters = {
		'clf__estimator__n_estimators': list(range(50,151,25)),
		'clf__estimator__max_features': ["sqrt","log2"]
	}

	# 12 jobs are used to utilize the multiple cores of the CPU. 
	# If it fails to execute try changing the number of jobs and run again. 
	# If it keeps failing, try to run the load_data_train_export_model() function from Jupyter Notebook
	# If it keeps failing, remove the n_jobs parameter to run the optimization on a single core
	cv_random_forest = GridSearchCV(estimator=pipeline, param_grid=RandomForest_parameters, verbose=3,n_jobs=12)
	return cv_random_forest

def train_and_test_model(model,X_train,y_train,X_test,y_test):
	model.fit(X_train, y_train)
	y_test_pred = model.predict(X_test)
	print_model_metrics(y_test_pred,y_test,y_test.columns.values)
	return model


def save_model(model,target_file):
	pickle.dump(model, open(target_file, 'wb'))


def load_data_train_export_model(database_filepath,model_filepath):
	df = load_database(database_filepath)

	# validation sets will be used to quickly test the fitting function for code errors. It can be otherwise ignored.
	X_train, X_valid, X_test, y_train, y_valid, y_test = get_train_valid_test_data(df)

	model = create_model()

	trainedModel = train_and_test_model(model,X_train,y_train,X_test,y_test)

	save_model(trainedModel,model_filepath)

def main():

	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]

		# the rest of the main function is refactored seperately to allow for external calls from Jupyter Notebook
		load_data_train_export_model(database_filepath,model_filepath)

	else:
		print('Please provide the paths of the source_database and target_model '\
			  'files as the first and second argument respectively'\
			  '\n\nExample: python train.py cleaned_messages_and_categories.db trained_model.pkl'\
			  )

if __name__ == '__main__':
	main()