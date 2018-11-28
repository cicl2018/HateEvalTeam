import gzip
import csv
import numpy as np
from keras import regularizers
from keras.layers import Dense, Dropout, Embedding, GRU
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

def get_data():
    file = 'C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#1 Practice\\trial_en.tsv'

    list_of_sentences = list()
    list_of_hateful = list()
    with open(file, encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            list_of_sentences.append(row[1])
            list_of_hateful.append(row[2])
        print(list_of_sentences)

    alphabet = dict()
    train_x = []

    for sent in list_of_sentences:
        word = []
        for char in sent:
            if char in alphabet:
                word.append(alphabet[char])
            else:
                alphabet[char] = len(alphabet) + 1
                word.append(alphabet[char])
            train_x.append(word)

    train_y = list_of_hateful
    #print(train_y)
    return train_x, train_y



def recurrent_networks():
    train_x, train_y = get_data()
    epoch = 10
    dropout = 0.5

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    recurrent_model = Sequential()
    recurrent_model.add(Embedding(input_dim=train_x.shape[1]))
    recurrent_model.add(Dropout(dropout))
    recurrent_model.add(GRU(units=256, activation='relu'))
    recurrent_model.add(Dense(units=1, activation='sigmoid'))
    recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    recurrent_model.fit(train_x, train_y, epochs=epoch, batch_size=32)

    score = recurrent_model.evaluate()
    

get_data()