Text Questions Classification

A simple text classification script using sklearn.feature_extraction for tokenize the data and scikit-learn naive_bayes for classification.

About the Model:

1_Model: question_classifier.py
Based on the data set size (small) and assumed the features are independent, I decided to implement Naive Bayes Model because is easy to implement, the classifier model is fast to build and the model can be modified with new training data without having to rebuild the model.

2_Model: doc2vec.py
Using doc2vec for sentence embeddings with some prediction model but because of data set size, the sentence embeddings with doc@vec is not going to provide a really good accuracy, based in my experience I could suppose the accuracy less than 40%, however with a large data set the accuracy can rise noticeably.

Running the classifier:
	- sudo pip install -r requirements.txt
	- python question_classifier.py, after running this line the classifier will require the question text, just type the text and press enter.

Prediction format:
	- 'Question Text' : 1-> INN.2, 2-> ALI.5
	- The classifier provides first matchs (1-> INN.2) and second matchs (2-> ALI.5).
