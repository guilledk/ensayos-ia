#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy
import pandas as pd

COLUMN_NAMES = [
        'SepalLength',
        'SepalWidth',
        'PetalLength',
        'PetalWidth',
        'Species'
]

FILEPATH_TRAINING = 'datasets/iris_training.csv'
FILEPATH_TEST = 'datasets/iris_test.csv'

# import datasets
training_dataset = pd.read_csv(
    FILEPATH_TRAINING,
    names=COLUMN_NAMES,
    header=0
)

train_x = training_dataset.iloc[:, 0:4]
train_y = training_dataset.iloc[:, 4]

test_dataset = pd.read_csv(
    FILEPATH_TEST,
    names=COLUMN_NAMES,
    header=0
)

test_x = test_dataset.iloc[:, 0:4]
test_y = test_dataset.iloc[:, 4]

# encode to output format
encoding_train_y = np_utils.to_categorical(train_y)
encoding_test_y = np_utils.to_categorical(test_y)

# create model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(train_x, encoding_train_y, epochs=300, batch_size=10)

scores = model.evaluate(test_x, encoding_test_y)
print("\nAccuracy: %.2f%%" % (scores[1]*100))
