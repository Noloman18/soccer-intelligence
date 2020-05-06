import sys
import os
import numpy as np

from tensorflow import keras

TRAINING_SET_PERCENTAGE = 0.8


def load_data(fp):
    csv_data = np.loadtxt(fp, dtype=np.int, skiprows=1, delimiter=',')

    np.random.shuffle(csv_data)

    data_length = len(csv_data)
    train_length = int(np.floor(data_length * TRAINING_SET_PERCENTAGE))

    train_data = csv_data[:train_length, :-1]
    train_label = csv_data[:train_length, -1]

    test_data = csv_data[train_length:, :-1]
    test_label = csv_data[train_length:, -1]

    return (train_data, train_label), (test_data, test_label)


def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=64, activation='relu', input_dim=510))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=3, activation='softmax'))
    return model


def accuracy(train_label, predictions, predicted_y):
    tp = np.sum(train_label == predicted_y)

    fp = 0.0
    fn = 0.0

    for idx in range(len(train_label)):
        if train_label[idx] != predicted_y and predictions[idx] == predicted_y:
            fp += 1
        elif train_label[idx] == predicted_y and predictions[idx] != predicted_y:
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return predicted_y, precision, recall


def train_model(model: keras.models.Sequential, train_data, train_label, test_data, test_label):
    train_label = keras.utils.to_categorical(train_label, num_classes=3)
    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=64, epochs=300, verbose=2)

    categorial_test_label = keras.utils.to_categorical(test_label, num_classes=3)
    loss_and_metrics = model.evaluate(test_data, categorial_test_label, batch_size=64, verbose=2)
    print(loss_and_metrics)

    predictions = model.predict(test_data, batch_size=64)
    predictions = np.argmax(predictions, axis=1)
    home_predictions = accuracy(test_label, predictions, 0)
    draw_predictions = accuracy(test_label, predictions, 1)
    away_predictions = accuracy(test_label, predictions, 2)

    print(home_predictions, draw_predictions, away_predictions)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("The application is expecting the location of the file to be provided as an argument")

    file_location = sys.argv[1]
    if not os.path.exists(file_location):
        raise Exception("File {} was not found".format(file_location))

    (train_data, train_label), (test_data, test_label) = load_data(file_location)

    model = create_model()

    train_model(model, train_data, train_label, test_data, test_label)

    print('finished')
