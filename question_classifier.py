import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle


     			   ########################
                   #                      #
                   # Questions Classifier #
                   #                      #
                   ########################



def clean_up_label(text):
    labels = []
    for label in text:
        try:
            words = label.split(',')
            labels.append(str(words[0].strip()))
        except:
            #Average Imputation for missing or wrong data. In this case the Average Imputation is TEA.2
            labels.append('TEA.2')

    return labels

def second_prediction_value(values):
    # Calculating the second prediction
    predictions = values
    values = np.delete(values,np.where(predictions == np.max(predictions))[1][0])
    v = max(values)
    return np.where(predictions == v)[1][0]

def preprocessing_Data():

    # Reading the dataSet
    data_set = pd.read_csv("labeled_data.csv",header= 0 ,encoding = "ISO-8859-1",delimiter='|')
    # Spliting the dataSet in training_set and testing_set
    x_train, x_test, y_train_1, y_test_1 = train_test_split(data_set.values[:, 0], data_set.values[:, 1], random_state=0, test_size=0.20)
    y_train = clean_up_label(y_train_1)
    y_test = clean_up_label(y_test_1)
    return x_train, x_test, y_train, y_test

def train_classifier(x_train,y_train,x_test,y_test):
    # Extracting text features
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Initializing Naives-Bayes Model and training
    model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True).fit(X_train_tfidf, y_train)

    # Saving the model
    f = open('classifier.pickle', 'wb')
    pickle.dump(model, f)
    f.close()

   # Testing the model
    X_test_counts = count_vect.transform(x_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    score = model.score(X_test_tfidf, y_test)
    print('Testing accuracy: ' + str(score * 100) + '%')

    return model

def prediction_classifier(document,model_path,x_train):
	# Initializing dictionary text features
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit_transform(X_train_counts)

    # Loading the model
    f = open(model_path, 'rb')
    model_classifier = pickle.load(f)
    f.close()

    # Extracting text features from input question
    document_counts = count_vect.transform(document)
    document_tfidf = tfidf_transformer.transform(document_counts)

    # Making predictions
    predictedn = model_classifier.predict(document_tfidf)
    prob = model_classifier.predict_proba(document_tfidf)

    # Second result
    second = second_prediction_value(prob)
    second_class = np.unique(y_train)[second]

    # Printing the prediction
    for doc, category in zip(document, predictedn):
        print('%r : 1-> %s, 2-> %s' % (doc, category,second_class))


    return 1


if __name__ == "__main__":

	# Input text question
	question = input("Please enter your text question: ")
	q = [str(question)]
	# Path to load the model
	model_path = 'classifier.pickle'
	# Read Data set
	x_train, x_test, y_train, y_test = preprocessing_Data()
	# 	Uncomment next code line if you want to train the model for yourself
	#	train_classifier(x_train, y_train,x_test,y_test)

	prediction_classifier(q, model_path, x_train)









