import csv
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, Flatten
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
import time


def get_data(train_file, test_file, padding='post'):
    list_of_sentences_train = list()
    list_of_hateful_train = list()
    list_of_aggressive_train = list()
    list_of_targeted_train = list()
    max_length = list()

    ### Train file ###
    with open(train_file, encoding="utf8") as cur_train_file:
        reader = csv.reader(cur_train_file, delimiter='\t')
        for row in reader:
            list_of_sentences_train.append(row[1])
            list_of_hateful_train.append(row[2])
            list_of_targeted_train.append(row[3])
            list_of_aggressive_train.append(row[4])

    list_of_sentences_train.pop(0)
    list_of_hateful_train.pop(0)
    list_of_targeted_train.pop(0)
    list_of_aggressive_train.pop(0)

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
    train_y_hateful = list_of_hateful_train
    train_y_targeted = list_of_targeted_train
    train_y_aggressive = list_of_aggressive_train

    train_x = pad_sequences(train_x, padding=padding, value=0, maxlen=longest_sent, truncating='post')

    ### TEST FIlE ###

    list_of_sentences_test = list()
    # list_of_hateful_test = list()
    # list_of_aggressive_test = list()
    # list_of_targeted_test = list()

    with open(test_file, encoding="utf8") as cur_test_file:
        reader = csv.reader(cur_test_file, delimiter='\t')
        for row in reader:
            list_of_sentences_test.append(row[1])

        list_of_sentences_test.pop(0)

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

        test_x = pad_sequences(test_x, padding=padding, value=0, maxlen=longest_sent, truncating='post')

        print(len(train_x), len(train_y_hateful), len(train_y_aggressive), len(train_y_targeted), len(test_x))

        return train_x, train_y_hateful, train_y_aggressive, train_y_targeted, test_x, longest_sent


time1 = time.ctime()


def recurrent_network_hateful_eval():
    train_x, train_y_hateful, train_y_aggressive, train_y_targeted, test_x, max_sent = get_data(
        train_file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\train_dev_en_merged.tsv',
        test_file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#3 Evaluation-English-A\\test_en.tsv',
        padding='post')

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y_hateful)
    train_y = np.reshape(train_y, (-1, 1))
    test_x = np.asarray(test_x)

    recurrent_model = Sequential()
    recurrent_model.add(Embedding(input_dim=5000, output_dim=28, input_length=max_sent, mask_zero=False))
    recurrent_model.add(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))
    recurrent_model.add(Flatten())
    recurrent_model.add(Dense(units=1, activation='sigmoid'))
    recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=5)
    recurrent_model.fit(train_x, train_y, callbacks=[es], epochs=50, batch_size=128)

    y_test_pred_hateful = recurrent_model.predict(test_x)

    return y_test_pred_hateful


def recurrent_network_aggressive_eval():
    print("--------------- AGGRESSSIVE -----------------")
    train_x, train_y_hateful, train_y_aggressive, train_y_targeted, test_x, max_sent = get_data(
        train_file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\train_dev_en_merged.tsv',
        test_file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#3 Evaluation-English-A\\test_en.tsv',
        padding='post')

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y_aggressive)
    train_y = np.reshape(train_y, (-1, 1))
    test_x = np.asarray(test_x)

    recurrent_model = Sequential()
    recurrent_model.add(Embedding(input_dim=5000, output_dim=28, input_length=max_sent, mask_zero=False))
    recurrent_model.add(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))
    recurrent_model.add(Flatten())
    recurrent_model.add(Dense(units=1, activation='sigmoid'))
    recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=5)
    recurrent_model.fit(train_x, train_y, callbacks=[es], epochs=50, batch_size=128)

    y_test_pred_aggressive = recurrent_model.predict(test_x)

    return y_test_pred_aggressive


def recurrent_network_targeted_eval():
    print("----------------- TARGETED ----------------")
    train_x, train_y_hateful, train_y_aggressive, train_y_targeted, test_x, max_sent = get_data(
        train_file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#2 Development-English-A\\train_dev_en_merged.tsv',
        test_file='C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#3 Evaluation-English-A\\test_en.tsv',
        padding='post')

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y_targeted)
    train_y = np.reshape(train_y, (-1, 1))
    test_x = np.asarray(test_x)

    recurrent_model = Sequential()
    recurrent_model.add(Embedding(input_dim=5000, output_dim=28, input_length=max_sent, mask_zero=False))
    recurrent_model.add(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))
    recurrent_model.add(Flatten())
    recurrent_model.add(Dense(units=1, activation='sigmoid'))
    recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=5)
    recurrent_model.fit(train_x, train_y, callbacks=[es], epochs=50, batch_size=128)

    y_test_pred_targeted = recurrent_model.predict(test_x)

    return y_test_pred_targeted


def create_file():
    test_file = 'C:\\Users\\mihai\\PycharmProjects\\SharedTaskHS\\HateEvalTeam\\Data Files\\Data Files\\#3 Evaluation-English-A\\test_en.tsv'
    y_test_pred_hateful = recurrent_network_hateful_eval()
    y_test_pred_aggressive = recurrent_network_aggressive_eval()
    y_test_pred_targeted = recurrent_network_targeted_eval()
    ids = list()

    with open(test_file, encoding="utf8") as file_test:
        reader = csv.reader(file_test, delimiter='\t')
        for row in reader:
            ids.append(row[0])
    ids.pop(0)
    file_test.close()

    smaller_than = 0.499
    list_of_numbers_hateful = list()
    list_of_numbers_aggressive = list()
    list_of_numbers_targeted = list()

    for x in y_test_pred_hateful:
        if x < smaller_than:
            list_of_numbers_hateful.append(int("0"))
        else:
            list_of_numbers_hateful.append(int("1"))

    for y in y_test_pred_aggressive:
        if y < smaller_than:
            list_of_numbers_aggressive.append(int("0"))
        else:
            list_of_numbers_aggressive.append(int("1"))

    for z in y_test_pred_targeted:
        if z < smaller_than:
            list_of_numbers_targeted.append(int("0"))
        else:
            list_of_numbers_targeted.append(int("1"))

    i = 0

    with open('en_b_comp.tsv', 'w') as final_file:
        while i < len(ids):
            final_file.write(
                str(ids[i]) + "\t" + str(list_of_numbers_hateful[i]) + "\t" + str(list_of_numbers_aggressive[i])
                + "\t" + str(list_of_numbers_targeted[i]) + "\n")
            i += 1

    final_file.close()

    j = 0
    with open('en_b_cagri.tsv', 'w') as final_file_2:
        while j < len(ids):
            final_file_2.write(
                str(ids[j]) + "\t" + str(list_of_numbers_aggressive[j]) + "\t" + str(list_of_numbers_hateful[j])
                + "\t" + str(list_of_numbers_targeted[j]) + "\n")

            j += 1

    final_file_2.close()


    #with open('en_a.tsv', 'w') as final_file:
        #while i < len(ids):
            #final_file.write(str(ids[i]) + "\t" + str(list_of_numbers_hateful[i]) + "\n")
            #i += 1
    #final_file.close()


create_file()
