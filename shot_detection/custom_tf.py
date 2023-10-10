import tensorflow as tf
import pandas as pd
from keras import layers, models
import numpy as np
import os
from random import sample
from tqdm import tqdm
import logging
from PIL import Image

tf.get_logger().setLevel(logging.ERROR)

height = 360
width = 640
channels = 3
frames = 3
batch_size = 5
DATASET_PATH = r'C:\Users\fziet\OneDrive\Pulpit\szkola\inzynierka\shot_detection\dataset\frames'
SHOTS_PATH = r'C:\Users\fziet\OneDrive\Pulpit\szkola\inzynierka\shot_detection\dataset\shots'


# Define a function to load and preprocess each image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image.load()
    return np.asarray(image.resize([width, height]), dtype="uint8")


df_labels = pd.read_csv('labels.csv', index_col=None)
shots_files = os.listdir(SHOTS_PATH)


sequences_true = []
sequences_false = []

sequence_labels = []

sciezki = df_labels['path'].tolist()
ile = len(sciezki)
for i, j in tqdm(enumerate(sciezki), total=ile):
    if i >= ile - 2:
        continue
    numer1 = j.split('_')[1]
    numer2 = sciezki[i + 1].split('_')[1]
    numer3 = sciezki[i + 2].split('_')[1]
    if numer1 == numer2 == numer3:
        picture_path = f"{DATASET_PATH}\\{j.replace('.txt', '.jpg')}"
        picture_path1 = f"{DATASET_PATH}\\{sciezki[i + 1].replace('.txt', '.jpg')}"
        picture_path2 = f"{DATASET_PATH}\\{sciezki[i + 2].replace('.txt', '.jpg')}"
        sequence = np.array([load_and_preprocess_image(picture_path),
                             load_and_preprocess_image(picture_path1),
                             load_and_preprocess_image(picture_path2)])
        if j in shots_files or sciezki[i+1] in shots_files or sciezki[i+2] in shots_files:
            sequences_true.append(sequence)
        else:
            sequences_false.append(sequence)


sequences = np.array(sequences_true + sample(sequences_false, len(sequences_true)))
sequence_labels = [1] * len(sequences_true) + [0] * len(sequences_true)
labels_reshaped = np.asarray(sequence_labels).astype('float32').reshape((-1, 1))

'''
sequences_dataset = tf.data.Dataset.from_tensor_slices((sequences, labels_reshaped))

sequences_dataset = sequences_dataset.shuffle(buffer_size=len(sequences))

train_size = int(0.8 * len(sequences))
train_dataset = sequences_dataset.take(train_size)
val_dataset = sequences_dataset.skip(train_size)
'''
model = models.Sequential([
    layers.Conv3D(32, (3, 3, 3), strides=(3, 3, 1), activation='relu', padding='same',
                  input_shape=(frames, height, width, channels)),
    layers.Conv3D(32, (3, 3, 3),  strides=(2, 2, 3), padding='same', activation='relu'),
    layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same'),

    layers.Conv3D(64, (3, 3, 3),  strides=(2, 2, 3), padding='same', activation='relu'),
    layers.Conv3D(64, (3, 3, 3),  strides=(2, 2, 3), padding='same', activation='relu'),
    layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same'),

    layers.Conv3D(64, (3, 3, 3),  strides=(3, 3, 1), padding='same', activation='relu'),
    layers.Conv3D(64, (3, 3, 3),  strides=(3, 3, 1), padding='same', activation='relu'),
    layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same'),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print('zkompilowano model')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, start_from_epoch=100)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best.hdf5', save_best_only=True)
model.fit(x=sequences, y=labels_reshaped, epochs=500, batch_size=5, validation_split=0.2,
          callbacks=[early_stopping, model_checkpoint], shuffle=True)
print('wytrenowano model')
model.save('custom_keras.keras')
print('zapisano model')
