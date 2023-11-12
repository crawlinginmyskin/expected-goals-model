import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
height = 240
width = 426
channels = 3


def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image.load()
    return np.asarray(image.resize([width, height]))


def preprocess_label_df() -> pd.DataFrame:
    df = pd.read_csv('xg_labels.csv')
    df = df[['image', 'number']]
    df = df.dropna()
    df['image'] = df['image'].apply(lambda x: x.split('/')[-1])
    numbers = []
    for n in df['number']:
        trimmed = f"0.{n.split('.')[1][:2]}"
        if trimmed[-1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            trimmed = trimmed[:-1]
        numbers.append(float(trimmed))

    df['number'] = numbers
    return df


def load_images_and_labels():
    dataset_path = r"C:\Users\fziet\OneDrive\Pulpit\szkola\inzynierka\shot_detection\dataset\frames"
    df = preprocess_label_df()
    images = []
    labels = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        images.append(load_and_preprocess_image(f"{dataset_path}\\{row['image']}"))
        labels.append(float(row['number']))
    return np.asarray(images), np.asarray(labels)
