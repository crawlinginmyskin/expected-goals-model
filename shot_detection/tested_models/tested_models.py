from keras.layers import TimeDistributed, Flatten, Dense, Conv2D, MaxPool2D, Input, Conv3D, MaxPool3D, Dropout, LSTM
from keras import models

height = 240
width = 426
channels = 3
frames = 5
batch_size = 2


def get_time_distributed2d():
    model = models.Sequential([
        Input(shape=(frames, height, width, channels)),
        TimeDistributed(Conv2D(32, (3, 3), padding='same', strides=(2,2), activation='relu')),
        TimeDistributed(Conv2D(32, (3, 3), padding='same', strides=(2,2), activation='relu')),
        TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))),
        TimeDistributed(Conv2D(16, (3, 3), padding='same', strides=(2, 2), activation='relu')),
        TimeDistributed(Conv2D(16, (3, 3), padding='same', strides=(2, 2), activation='relu')),
        TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))),
        TimeDistributed(Dense(128, activation='relu')),
        TimeDistributed(Dense(64, activation='relu')),
        TimeDistributed(Dense(32, activation='relu')),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='sigmoid')
    ])
    return model


def get_3dconv():
    model = models.Sequential([
        Conv3D(32, (3, 3, 3), strides=(3, 3, 1), activation='relu', padding='same',
                      input_shape=(frames, height, width, channels)),
        Conv3D(32, (3, 3, 3), strides=(2, 2, 3), padding='same', activation='relu'),
        MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same'),

        Conv3D(64, (3, 3, 3), strides=(2, 2, 3), padding='same', activation='relu'),
        Conv3D(64, (3, 3, 3), strides=(2, 2, 3), padding='same', activation='relu'),
        MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same'),

        Conv3D(64, (3, 3, 3), strides=(3, 3, 1), padding='same', activation='relu'),
        Conv3D(64, (3, 3, 3), strides=(3, 3, 1), padding='same', activation='relu'),
        MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same'),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model


def get_lstm():
    model = models.Sequential([
        Input(shape=(frames, height, width, channels)),
        TimeDistributed(Conv2D(32, (3, 3), padding='same', strides=(2,2), activation='relu')),
        TimeDistributed(Conv2D(32, (3, 3), padding='same', strides=(2,2), activation='relu')),
        TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))),
        TimeDistributed(Conv2D(16, (3, 3), padding='same', strides=(2, 2), activation='relu')),
        TimeDistributed(Conv2D(16, (3, 3), padding='same', strides=(2, 2), activation='relu')),
        TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))),
        TimeDistributed(Flatten()),
        LSTM(16, return_sequences=False, activation='relu'),
        Dense(256, activation='relu'),
        Dense(2, activation='sigmoid')
    ])
    return model
