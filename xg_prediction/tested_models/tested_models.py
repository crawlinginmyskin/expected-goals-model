from keras.layers import TimeDistributed, Flatten, Dense, Conv2D, MaxPool2D, Input, Conv3D, MaxPool3D, Dropout, LSTM
from keras import models

height = 240
width = 426
channels = 3


def get_conv2d_model():
    model = models.Sequential([
        Input(shape=(height, width, channels)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model
