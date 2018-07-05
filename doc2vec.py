
import gensim
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.models import doc2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import keras as kr
from keras.layers import Dense,Dropout
import numpy

from keras.optimizers import Adam

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


def clean_up_text(text):

    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", text)
    # Convert words to lower case and split them
    words = review_text.lower().split()
    # Remove stopwords
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return words

def clean_up_label(text):
    labels = []
    for label in text:
        try:
            words = label.split(',')
            labels.append(str(words[0].strip()))
        except:
            print('Error')
            print(label)
            labels.append('TEA.2')

    return labels



def preprocessing_Data():
    data_set = pd.read_csv("/Users/Victor/Downloads/machine_learning_engineer_coding_test/labeled_data.csv", encoding = "ISO-8859-1",delimiter='|')
    x_train, x_test, y_train_1, y_test_1 = train_test_split(data_set.values[:, 0], data_set.values[:, 1], random_state=1, test_size=0.21)
    data_full = x_train.tolist() + x_test.tolist()
    x_train= labeledLineSentence(x_train, 'Train')
    x_test = labeledLineSentence(x_test, 'Test')
    y_train = clean_up_label(y_train_1)
    y_test = clean_up_label(y_test_1)
    all = labeledLineSentence(data_full, 'All')
    return x_train, x_test, y_train, y_test, all

def preprocessing_Data_RNN():
    data_set = pd.read_csv("/Users/Victor/Downloads/machine_learning_engineer_coding_test/labeled_data.csv",
                           encoding="ISO-8859-1", delimiter='|')
    x_train, x_test, y_train_1, y_test_1 = train_test_split(data_set.values[:, 0], data_set.values[:, 1],
                                                            random_state=0, test_size=0.15)
    data_full = x_train.tolist() + x_test.tolist()
    y_data_full = data_set.values[:, 1]
    x_train = labeledLineSentence(x_train, 'Train')
    x_test = labeledLineSentence(x_test, 'Test')
    y_train_2 = clean_up_label(y_train_1)
    y_train = encode_(y_train_2)
    y_test_2 = clean_up_label(y_test_1)
    y_test = encode_(y_test_2)
    all = labeledLineSentence(data_full, 'All')
    return x_train, x_test, y_train, y_test, all, y_data_full

LabeledSentence = gensim.models.doc2vec.TaggedDocument
label_encoder = LabelEncoder()

def encode_(Y):
    label_encoder.fit(Y)
    encoded_Y = label_encoder.transform(Y)
    print('Encode')
    print(encoded_Y)
    return encoded_Y

def labeledLineSentence(source,label):
    sentences = []
    label_name = label
    for i, v in enumerate(source):
        label = label_name + '_' + str(i)
        sentences.append(LabeledSentence(clean_up_text(v), [label]))
    return sentences

def trainning_doc2vec(source,training_vectors):
    print('Building Doc2vec')
    d2v = doc2vec.Doc2Vec(min_count=1, window=2, vector_size=100, sample=1e-4, workers=8)
    d2v.build_vocab(source)
    print('Training Doc2vec')
    d2v.train(training_vectors, total_examples=len(training_vectors), epochs=25)
    print('Saving Model')
    # d2v.save('./imdb.d2v')
    return d2v

def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type,):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        index = i
        if vectors_type == 'Test':
            index += vectors_size
        prefix = 'All_' + str(index)
        vectors[i] = doc2vec_model.docvecs[prefix]
        print('extracting vectors')
        print(len(vectors))
    return vectors

def train_classifier(train_vectors, training_labels):

    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_vectors, np.array(training_labels))
    return model

def train_classifier_k(train_vectors,test_vectors,Y,trai_encoded_Y,test_encoded_Y):

    values = Y.ravel()
    dummy_y = kr.utils.np_utils.to_categorical(trai_encoded_Y)
    dummy_y_t = kr.utils.np_utils.to_categorical(test_encoded_Y)
    print(dummy_y)
    out_layer = 4

    cvscores = []
    model = kr.Sequential()
    model.add(Dense( input_dim=100,output_dim=500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=1200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=600, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=int(out_layer), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    with open('bgModel_architecture.json', 'w') as f:
        f.write(model.to_json())
    model.fit(train_vectors, dummy_y, validation_data=(test_vectors, dummy_y_t), batch_size=50, nb_epoch=25)
    model.save_weights('bgweights.h5')
    scores = model.evaluate(test_vectors, dummy_y_t, batch_size=32)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)


    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    print('score')
    print(scores)
    data = []
    vec = clean_up_text('My team holds ourselves accountable for results')
    data.append(vec)
    test_1_vec = get_vectors(d2v, len(data), 100, 'Test')
    print('Prediction')
    p = model.predict_classes(test_1_vec)
    print(model.predict_classes(test_1_vec))
    print(label_encoder.inverse_transform(p))

    return scores

def test_classifier(classifier, test_vectors, testing_labels):
    print("Train Doc2Vec on testing set")
    testing_predictions = classifier.predict(test_vectors)
    print(testing_predictions)
    print('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    print('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))
    print(classifier.score(test_vectors, testing_labels))



x_train, x_test, y_train, y_test, all = preprocessing_Data()
d2v = trainning_doc2vec(all,x_train)


train_vectors = get_vectors(d2v, len(x_train), 100, 'Train')
test_vectors = get_vectors(d2v, len(x_test), 100, 'Test')
#
model_1 = train_classifier(train_vectors, y_train)
test_classifier(model_1, test_vectors, y_test)

data = []
vec = clean_up_text('I knew what was expected of me to be successful in my role')
data.append(vec)
test_1_vec = get_vectors(d2v, len(data), 100, 'Test')
print('Prediction')
print(model_1.predict(test_1_vec))
print(model_1.predict_proba(test_1_vec))




