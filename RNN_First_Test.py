import gzip
import csv
import numpy as np
from keras import regularizers
from keras.layers import Dense, Dropout, Embedding, GRU
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
import keras_metrics

def get_data(file, padding='post'):
    #file = 'C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#1 Practice\\trial_en.tsv'

    list_of_sentences = list()
    list_of_hateful = list()
    max_length = list()
    counter = 0

    with open(file, encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            list_of_sentences.append(row[1])
            list_of_hateful.append(row[2])

    for x in list_of_sentences:
        max_length.append(len(x))
    longest_sent = max(max_length)

    alphabet = dict()
    train_words = []

    for sent in list_of_sentences:
        word = []
        for char in sent:
            if char in alphabet:
                word.append(alphabet[char])
            else:
                alphabet[char] = len(alphabet) + 1
                word.append(alphabet[char])
            train_words.append(word)

    train_x = train_words[:500]
    test_x = train_words[500:]

    list_of_hateful = list(map(int, list_of_hateful[1:]))
    train_y = list_of_hateful[:500]
    test_y = list_of_hateful[500:]

    train_x = pad_sequences(train_x, padding=padding, value=0, maxlen=longest_sent, truncating='pre')
    test_x = pad_sequences(train_x, padding=padding, value=0, maxlen=longest_sent, truncating='pre')

    print(type(list_of_hateful[2]))
    print(train_x[1], train_y[1])
    print(train_y[0], train_y[-1])
    print(len(train_y),len(test_y))
    return train_x, train_y, test_x, test_y

def recurrent_networks():
    train_x, train_y, test_x, test_y = get_data(file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\dev_en.tsv', padding='post')
    epoch = 1
    dropout = 0.5

    #print(type(train_x),type(train_y),type(test_x),type(test_y))
    #train_x = np.array(train_x)
    train_y = np.array(train_y)
    #test_x = np.array(test_x)
    test_y = np.array(test_y)

    #print(type(train_x), type(train_y), type(test_x), type(test_y))

    recurrent_model = Sequential()
    recurrent_model.add(Embedding(input_dim=128, output_dim=128))
    recurrent_model.add(Dropout(dropout))
    recurrent_model.add(GRU(units=256, activation='relu'))
    recurrent_model.add(Dense(units=1, activation='sigmoid'))
    recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    recurrent_model.fit(train_x, train_y, epochs=epoch, batch_size=32)

    score = recurrent_model.evaluate(test_x, test_y)
    print(score)

#get_data(file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\dev_en.tsv')
recurrent_networks()