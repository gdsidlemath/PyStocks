import datetime
import pandas as pds
import time
import warnings
import numpy as np
import tensorflow as tf
import lstm
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import csv
import urllib.request

tf.python.control_flow_ops = tf

map_fn = tf.map_fn

aa = np.asarray

warnings.filterwarnings("ignore")

def getClosingData(symbol, yrrange):
    now = datetime.datetime.now()

    day = str(now.day - 1)
    month = str(now.month - 1)
    yr = str(now.year - yrrange)

    url = 'http://chart.finance.yahoo.com/table.csv?s=' + symbol + '&a=' + month + '&b=' + day + '&c=' + yr + '&d=' + month + '&e=' + day + '&f=' + str(
        now.year) + '&g=d&ignore=.csv'

    test = pds.read_csv(url)

    test = test[::-1]

    x = pds.to_datetime(test.Date)

    Close = aa(test['Adj Close'])

    return Close, x

def proc_data(data, seq_len, normalise_window):

    sequence_length = seq_len + 1
    result = []

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

yrrange = 5
symbol = '^GSPC'

data, days = getClosingData(symbol, yrrange)

window_length = 25

X_train, y_train, X_test, y_test = proc_data(data, window_length, True)

model = Sequential()

model.add(LSTM(
    input_shape=(window_length,1,),
    output_dim=int(window_length*.5),
    return_sequences=True))
model.add(Dropout(0.25))

model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.25))

model.add(Dense(output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time : ', time.time() - start)

model.fit(
    X_train,
    y_train,
    batch_size=100,
    nb_epoch=10,
    validation_split=0.1)

predictions = lstm.predict_sequence_full(model, X_test, window_length)
#predictions = lstm.predict_sequences_multiple(model, X_test, window_length, window_length)
#lstm.plot_results_multiple(predictions, y_test, window_length)

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)

ax.plot(predictions)
ax.plot(y_test)

plt.show()
