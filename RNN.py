import csv

import numpy
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, GRU, SimpleRNN, RNN, LSTM, Flatten, Activation
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
# from baseline import bong
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import time


def get_data(train_file, test_file, padding='post'):
    list_of_sentences_train = list()
    list_of_hateful_train = list()
    max_length = list()

    ### Train file ###
    with open(train_file, encoding="utf8") as cur_train_file:
        reader = csv.reader(cur_train_file, delimiter='\t')
        for row in reader:
            list_of_sentences_train.append(row[1])
            list_of_hateful_train.append(row[2])

    list_of_sentences_train.pop(0)
    list_of_hateful_train.pop(0)

    for x in list_of_sentences_train:
        max_length.append(len(x))
    longest_sent = max(max_length)

    alphabet = dict()
    train_words = []

    for sent in list_of_sentences_train:
        words = []
        for char in sent:
            if char in alphabet:
                words.append(alphabet[char])
            else:
                alphabet[char] = len(alphabet) + 1
                words.append(alphabet[char])

        train_words.append(words)

    train_x = train_words
    train_y = list_of_hateful_train

    train_x = pad_sequences(train_x, padding=padding, value=0, maxlen=longest_sent, truncating='post')

    ### TEST FIlE ###

    list_of_sentences_test = list()
    list_of_hateful_test = list()

    with open(test_file, encoding="utf8") as cur_test_file:
        reader = csv.reader(cur_test_file, delimiter='\t')
        for row in reader:
            list_of_sentences_test.append(row[1])
            list_of_hateful_test.append(row[2])

        list_of_sentences_test.pop(0)
        list_of_hateful_test.pop(0)

        test_words = []
        alphabet_length = len(alphabet)

        for test_sent in list_of_sentences_test:
            words = []
            for char_test in test_sent:
                if char_test in alphabet:
                    words.append(alphabet[char_test])
                else:
                    unk = alphabet_length + 1
                    words.append(unk)

            test_words.append(words)

        test_x = test_words
        test_y = list_of_hateful_test

        test_x = pad_sequences(test_x, padding=padding, value=0, maxlen=longest_sent, truncating='post')

        print(len(train_x), len(train_y), len(test_x), len(test_y))

        return train_x, train_y, test_x, test_y, longest_sent

# Train and test




# baseline model

time1 = time.ctime()

seed = 7
np.random.seed(seed)

def recurrent_network():

    train_x, train_y, test_x, test_y, max_sent = get_data(
        train_file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\train_en.tsv',
        test_file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\dev_en.tsv',
        padding='post')

    train_x = np.asarray(train_x)
    # train_x = train_x[:, :, np.newaxis]

    train_y = np.asarray(train_y)
    train_y = np.reshape(train_y, (-1, 1))

    test_x = np.asarray(test_x)
    # test_x = test_x[:, :, np.newaxis]

    test_y = np.asarray(test_y)
    test_y = np.reshape(test_y, (-1, 1))

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []

    for train, test in kfold.split(train_x, train_y):

        recurrent_model = Sequential()
        recurrent_model.add(Embedding(input_dim=5000, output_dim=28, input_length=max_sent, mask_zero=False))
        recurrent_model.add(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))
        #recurrent_model.add(LSTM(units=64, return_sequences=True, recurrent_dropout=0.2))
        # recurrent_model.add(Dropout(0.2))
        recurrent_model.add(Flatten())

        # recurrent_model.add(Dense(units=50, input_dim=2, activation='relu'))
        recurrent_model.add(Dense(units=1, activation='sigmoid'))
    
        recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Set callback functions to early stop training and save the best model so far
        es = EarlyStopping(monitor='val_loss', patience=5)

        # Train neural network
        # recurrent_model.fit(train_x, train_y, validation_split=0.2, epochs=100, callbacks=[es], batch_size=64)
        recurrent_model.fit(train_x[train], train_y[train], callbacks=[es], epochs=30, batch_size=128)
        scores = recurrent_model.evaluate(test_x, test_y)
        print("%s: %.2f%%" % (recurrent_model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    y_test_pred = recurrent_model.predict(test_x)

    smaller_than = 0.499
    list_of_numbers = list()

    for y in y_test_pred:
        #print(int(y))
        if y <= smaller_than:
            list_of_numbers.append(int("0"))
        else:
            list_of_numbers.append(int("1"))

    #print(list_of_numbers)

    test_y_new = []
    for x in test_y:
        x_new = float(x)
        x_new = np.asarray(x_new)
        test_y_new.append(x_new)

    prec = precision_score(test_y_new, y_test_pred.round(), average='macro')
    rec = recall_score(test_y_new, y_test_pred.round(), average='macro')
    f1 = f1_score(test_y_new, y_test_pred.round(), average='macro')

    time2 = time.ctime()

    print("Precision:", prec, "\n Recall:", rec, "\n F1-score:", f1)
    print(time1 + '\n' + time2)

    return y_test_pred


def create_file():
    test_file = 'C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\dev_en.tsv'
    y_test_pred = recurrent_network()
    ids = list()

    with open(test_file, encoding="utf8") as file_test:
        reader = csv.reader(file_test, delimiter='\t')
        for row in reader:
            ids.append(row[0])
    ids.pop(0)
    file_test.close()

    smaller_than = 0.499
    list_of_numbers = list()

    for y in y_test_pred:
        print(int(y))
        if y < smaller_than:
            list_of_numbers.append(int("0"))
        else:
            list_of_numbers.append(int("1"))

    print(list_of_numbers)

    i = 0

    with open('en_a.tsv', 'w') as final_file:
        #for x in ids, list_of_numbers:
        while i <= len(ids):
            #for y in list_of_numbers:
            final_file.write(str(ids[i]) + "\t" + str(list_of_numbers[i]) + "\n")
                #print(x, "---", y_new)
            i += 1
    final_file.close()


#recurrent_network()
create_file()