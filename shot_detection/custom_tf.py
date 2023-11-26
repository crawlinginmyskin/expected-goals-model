import tensorflow as tf
import pandas as pd
from keras.utils import to_categorical, Sequence
import numpy as np
import os
from random import sample
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from tested_models.tested_models import get_time_distributed2d, get_lstm, get_3dconv
import datetime

height = 240
width = 426
channels = 3
frames = 5
batch_size = 2
DATASET_PATH = r'C:\Users\fziet\OneDrive\Pulpit\szkola\inzynierka\shot_detection\dataset\frames'
SHOTS_PATH = r'C:\Users\fziet\OneDrive\Pulpit\szkola\inzynierka\shot_detection\dataset\shots'
HARD_NEGATIVES_PATH = r'C:\Users\fziet\OneDrive\Pulpit\szkola\inzynierka\shot_detection\dataset\negatives'


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


# Define a function to load and preprocess each image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image.load()
    return np.asarray(image.resize([width, height]), dtype="bfloat16")


df_labels = pd.read_csv('labels.csv', index_col=None)
shots_files = os.listdir(SHOTS_PATH)


sequences_true = []
sequences_false = []

sequence_labels = []

sciezki = df_labels['path'].tolist()
#sciezki = sciezki[:500]  # selecting small batch to overfit
ile = len(sciezki)
for i, j in tqdm(enumerate(sciezki), total=ile):
    if i >= ile - 5:
        continue
    numer1 = j.split('_')[1]
    numer2 = sciezki[i + 1].split('_')[1]
    numer3 = sciezki[i + 2].split('_')[1]
    numer4 = sciezki[i + 3].split('_')[1]
    numer5 = sciezki[i + 4].split('_')[1]
    if numer1 == numer2 == numer3 == numer4 == numer5:
        # this could be somehow parametrized
        frame1 = j.replace('.txt', '.jpg')
        frame2 = sciezki[i + 1].replace('.txt', '.jpg')
        frame3 = sciezki[i + 2].replace('.txt', '.jpg')
        frame4 = sciezki[i + 3].replace('.txt', '.jpg')
        frame5 = sciezki[i + 4].replace('.txt', '.jpg')
        picture_path = DATASET_PATH + "\\" + frame1
        picture_path1 = DATASET_PATH + "\\" + frame2
        picture_path2 = DATASET_PATH + "\\" + frame3
        picture_path3 = DATASET_PATH + "\\" + frame4
        picture_path4 = DATASET_PATH + "\\" + frame5

        sequence = [picture_path, picture_path1, picture_path2, picture_path3, picture_path4]
        suma = 0
        for m in range(0, 5):  # if 2 out of 5 frames are a shot, we label the sequence as a shot
            if sciezki[i+m] in shots_files:
                suma += 1
        if suma >= 2:
            sequences_true.append(sequence)
        else:
            sequences_false.append(sequence)

hard_negatives = os.listdir(HARD_NEGATIVES_PATH)
ile_nie = len(sequences_true)
sequences_hard_negatives = []
already_picked = []

print("preparing hard negatives: ")
for i in tqdm(range(ile_nie)):
    temp_sequence = sample(hard_negatives, frames)
    sequences_hard_negatives.append([fr'{HARD_NEGATIVES_PATH}\{photo}' for photo in temp_sequence])

sequences = np.array(sequences_true + sample(sequences_false, ile_nie) + sequences_hard_negatives)
sequences_photos = []
print("loading and processing images: ")
for sequence in tqdm(sequences):
    sequences_photos.append([load_and_preprocess_image(path) for path in sequence])
sequences_photos = np.asarray(sequences_photos)
sequence_labels = np.array([1] * len(sequences_true) + [0] * ile_nie + [0] * ile_nie)
labels_reshaped = to_categorical(sequence_labels)

model = get_time_distributed2d()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print('model was compiled')

x_train, x_test, y_train, y_test = train_test_split(sequences_photos, labels_reshaped, test_size=0.2, random_state=42)

train_gen = DataGenerator(x_train, y_train, 3)
test_gen = DataGenerator(x_test, y_test, 3)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('model_checkpoints/time_distributed_2d_2023-01-11.hdf5', save_best_only=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(train_gen, epochs=50, callbacks=[model_checkpoint], validation_data=test_gen)
print('wytrenowano model')

