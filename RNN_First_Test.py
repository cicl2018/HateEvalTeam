import csv
import numpy as np
from keras import regularizers
from keras.layers import Dense, Dropout, Embedding, GRU, SimpleRNN, RNN, LSTM, Flatten, Activation
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder



def get_data(file, padding='post'):
    list_of_sentences = list()
    list_of_hateful = list()
    max_length = list()
    counter = 0
    punctuation='@/\,;.:!?$123456789¿¡'
    table=str.maketrans(dict.fromkeys(punctuation))

    with open(file, encoding="utf8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            row1=row[1].lower()
            list_of_sentences.append(row1.translate(table))
            list_of_hateful.append(row[2])

    list_of_sentences.pop(0)
    list_of_hateful.pop(0)

    for x in list_of_sentences:
        max_length.append(len(x))
    longest_sent = max(max_length)

    alphabet = dict()
    train_words = []

    for sent in list_of_sentences:
        words = []
        for char in sent:
            if char in alphabet:
                words.append(alphabet[char])
            else:
                alphabet[char] = len(alphabet) + 1
                words.append(alphabet[char])

        train_words.append(words)

    train_x = train_words[:250]
    test_x = train_words[250:]

    list_of_hateful = list(map(int, list_of_hateful[1:]))
    train_y = list_of_hateful[:250]
    test_y = list_of_hateful[249:]

    train_x = pad_sequences(train_x, padding=padding, value=0, maxlen=longest_sent, truncating='pre')
    test_x = pad_sequences(test_x, padding=padding, value=0, maxlen=longest_sent, truncating='pre')

    # print(train_x[1], train_y[1])
    # print(len(train_x[-1]), len(test_x[0]))
    # print(len(train_y),len(test_y))

    print(len(list_of_sentences[1]), len(list_of_sentences[2]), len(list_of_sentences[3]))
    print((len(train_words[1])), len((train_words[2])), len((train_words[3])))
    print(len(train_x[1]), len(train_x[2]), len(train_x[3]))

    # print(train_x[1], test_x[1])
    # print(len(train_x[1]), len(test_x[1]))
    # print(len(train_x), len(test_x))
    return train_x, train_y, test_x, test_y


def recurrent_networks():
    train_x, train_y, test_x, test_y = get_data(
        file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 '
             'Development-English-A\\dev_en.tsv',
        padding='post')
    # epoch = 10
    # dropout = 0.1

    # print(type(train_x),type(train_y),type(test_x),type(test_y))
    train_x = np.asarray(train_x)
    train_x = train_x[:, :, np.newaxis]
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_x = test_x[:, :, np.newaxis]
    test_y = np.asarray(test_y)

    ##### EPOCHS & DROPOUT #####
    epochs = list(range(1, 8))
    dropout = [0.1, 0.2]

    # print(train_x.shape, train_y.shape)
    with open('results_LSTM.txt', 'w') as result_file:
        for dp in dropout:
            recurrent_model = Sequential()
            #recurrent_model.add(Embedding(input_dim=train_x.shape[1], output_dim=train_x.shape[1], input_shape=(851,)))
            # recurrent_model.add(Dropout(dropout))
            # recurrent_model.add()
            # recurrent_model.add(GRU(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
            # recurrent_model.add(Dropout(dp))
            recurrent_model.add(LSTM(units=64, activation='tanh', return_sequences=True))
            recurrent_model.add(LSTM(units=64, activation='tanh', return_sequences=True))
            #recurrent_model.add(LSTM(units=64, activation='tanh', return_sequences=True))
            recurrent_model.add(Dropout(dp))
            recurrent_model.add(Flatten())
            recurrent_model.add(Dense(units=1, activation='sigmoid'))
            recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            for epoch in epochs:
                recurrent_model.fit(train_x, train_y, epochs=epoch)
                score = recurrent_model.evaluate(test_x, test_y)

                y_test_pred = recurrent_model.predict(test_x)
                prec = precision_score(test_y, y_test_pred.round(), average='macro')
                rec = recall_score(test_y, y_test_pred.round(), average='macro')
                f1 = f1_score(test_y, y_test_pred.round(), average='macro')

                print("Dropout:", dp, "Epoch:", epoch, "F1:", f1)
                result_file.write("Dropout: " + str(dp) + ' --- ' + 'Epoch: ' + str(epoch) + ' --- ' + 'F1: ' + str(
                    f1) + " --- " + 'Precision:' + str(prec) + '\n')

    # print("Precision:", prec, "\n Recall:", rec, "\n F1-score:", f1)
    result_file.close()


time_1 = time.asctime(time.localtime(time.time()))


def recurrent_networks_2():
    train_x, train_y, test_x, test_y = get_data(
        file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\dev_en.tsv',
        padding='post')
    # epoch = 10
    # dropout = 0.1

    # print(type(train_x),type(train_y),type(test_x),type(test_y))
    train_x = np.asarray(train_x)
    #train_x = train_x[:, :, np.newaxis]
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    #test_x = test_x[:, :, np.newaxis]
    test_y = np.asarray(test_y)

    ##### EPOCHS & DROPOUT #####
    epochs = list(range(1, 31))
    dropout = [0.1, 0.2]

    # print(train_x.shape, train_y.shape)
    with open('results_2.txt', 'w') as result_file:
        for dp in dropout:
            recurrent_model = Sequential()
            recurrent_model.add(Embedding(input_dim=train_x.shape[1], output_dim=train_x.shape[1]))
            # recurrent_model.add(Dropout(dropout))
            # recurrent_model.add()
            recurrent_model.add(GRU(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
            recurrent_model.add(Dropout(dp))
            recurrent_model.add(Dense(units=1, activation='sigmoid'))
            recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            for epoch in epochs:
                recurrent_model.fit(train_x, train_y, epochs=epoch)
                score = recurrent_model.evaluate(test_x, test_y)

                y_test_pred = recurrent_model.predict(test_x)
                prec = precision_score(test_y, y_test_pred.round(), average='macro')
                rec = recall_score(test_y, y_test_pred.round(), average='macro')
                f1 = f1_score(test_y, y_test_pred.round(), average='macro')

                print("Dropout:", dp, "Epoch:", epoch, "F1:", f1)
                result_file.write("Dropout: " + str(dp) + ' --- ' + 'Epoch: ' + str(epoch) + ' --- ' + 'F1: ' + str(
                    f1) + " --- " + 'Score: ' + str(score[1]) + '\n')

    # print("Precision:", prec, "\n Recall:", rec, "\n F1-score:", f1)
    result_file.close()


def recurrent_networks_3():
    train_x, train_y, test_x, test_y = get_data(
        file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\dev_en.tsv',
        padding='post')
    # epoch = 10
    # dropout = 0.1

    # print(type(train_x),type(train_y),type(test_x),type(test_y))
    train_x = np.asarray(train_x)
    #train_x = train_x[:, :, np.newaxis]
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
   # test_x = test_x[:, :, np.newaxis]
    test_y = np.asarray(test_y)

    ##### EPOCHS & DROPOUT #####
    epochs = list(range(1, 31))
    dropout = [0.1, 0.2]

    # print(train_x.shape, train_y.shape)
    with open('results_3.txt', 'w') as result_file:
        for dp in dropout:
            recurrent_model = Sequential()
            recurrent_model.add(Embedding(input_dim=train_x.shape[1], output_dim=train_x.shape[1]))
            recurrent_model.add(Dropout(float(dp)))
            # recurrent_model.add()
            recurrent_model.add(GRU(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
            recurrent_model.add(Dropout(dp))
            recurrent_model.add(Dense(units=1, activation='sigmoid'))
            recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            for epoch in epochs:
                recurrent_model.fit(train_x, train_y, epochs=epoch)
                score = recurrent_model.evaluate(test_x, test_y)

                y_test_pred = recurrent_model.predict(test_x)
                prec = precision_score(test_y, y_test_pred.round(), average='macro')
                rec = recall_score(test_y, y_test_pred.round(), average='macro')
                f1 = f1_score(test_y, y_test_pred.round(), average='macro')

                print("Dropout:", dp, "Epoch:", epoch, "F1:", f1)
                result_file.write("Dropout: " + str(dp) + ' --- ' + 'Epoch: ' + str(epoch) + ' --- ' + 'F1: ' + str(
                    f1) + " --- " + 'Score: ' + str(score[1]) + '\n')

    # print("Precision:", prec, "\n Recall:", rec, "\n F1-score:", f1)
    result_file.close()


def recurrent_networks_4():
    train_x, train_y, test_x, test_y = get_data(
        file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\dev_en.tsv',
        padding='post')

    # print(type(train_x),type(train_y),type(test_x),type(test_y))
    train_x = np.asarray(train_x)
    train_x = train_x[:, :, np.newaxis]
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_x = test_x[:, :, np.newaxis]
    test_y = np.asarray(test_y)

    ##### EPOCHS & DROPOUT #####
    epochs = list(range(1, 31))
    dropout = [0.1, 0.2]

    with open('results_4.txt', 'w') as result_file:
        for dp in dropout:
            recurrent_model = Sequential()
            recurrent_model.add(GRU(units=32, activation='relu', return_sequences=True))
            # recurrent_model.add(Dropout(dropout))
            # recurrent_model.add()
            recurrent_model.add(GRU(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
            recurrent_model.add(Dropout(dp))
            recurrent_model.add(Dense(units=1, activation='sigmoid'))
            recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            for epoch in epochs:
                recurrent_model.fit(train_x, train_y, epochs=epoch)
                score = recurrent_model.evaluate(test_x, test_y)

                y_test_pred = recurrent_model.predict(test_x)
                prec = precision_score(test_y, y_test_pred.round(), average='macro')
                rec = recall_score(test_y, y_test_pred.round(), average='macro')
                f1 = f1_score(test_y, y_test_pred.round(), average='macro')

                print("Dropout:", dp, "Epoch:", epoch, "F1:", f1)
                result_file.write("Dropout: " + str(dp) + ' --- ' + 'Epoch: ' + str(epoch) + ' --- ' + 'F1: ' + str(
                    f1) + " --- " + 'Score: ' + str(score[1]) + '\n')

    result_file.close()


seed = 7
np.random.seed(seed)

def recurrent_networks_5():
    train_x, train_y, test_x, test_y = get_data(
        file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\dev_en.tsv',
        padding='post')

    # print(type(train_x),type(train_y),type(test_x),type(test_y))
    train_x = np.asarray(train_x)
    train_x = train_x[:, :, np.newaxis]
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_x = test_x[:, :, np.newaxis]
    test_y = np.asarray(test_y)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []

    for train, test in kfold.split(train_x, train_y):

        recurrent_model = Sequential()
        recurrent_model.add(LSTM(units=64, return_sequences=True, use_bias=True))
        recurrent_model.add(LSTM(units=64, return_sequences=True, use_bias=True))
        #recurrent_model.add(Dropout(0.2))
        #recurrent_model.add(Activation('tanh'))
        recurrent_model.add(Flatten())
        recurrent_model.add(Dense(units=1, activation='sigmoid'))
        recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        recurrent_model.fit(train_x[train], train_y[train], epochs=5)

        scores = recurrent_model.evaluate(test_x, test_y)
        print("%s: %.2f%%" % (recurrent_model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    y_test_pred = recurrent_model.predict(test_x)
    prec = precision_score(test_y, y_test_pred.round(), average='macro')
    rec = recall_score(test_y, y_test_pred.round(), average='macro')
    f1 = f1_score(test_y, y_test_pred.round(), average='macro')

    print("Precision:", prec, "\n Recall:", rec, "\n F1-score:", f1)




#recurrent_networks()
#recurrent_networks_2()
#recurrent_networks_3()
#recurrent_networks_4()
recurrent_networks_5()

time_2 = time.asctime(time.localtime(time.time()))
print(time_1 + '\n' + time_2)
