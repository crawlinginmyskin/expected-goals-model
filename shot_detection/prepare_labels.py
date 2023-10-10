import pandas as pd
import os

FRAMES_PER_SHOT = 27


def format_number(x):
    if x < 10:
        return f'00{x}'
    if x < 100:
        return f'0{x}'
    return f'{x}'


def load_data(frames_path='dataset/frames', shots_path='dataset/shots'):
    clips = os.listdir(frames_path)
    shots = os.listdir(shots_path)
    print(shots)
    picture_string = 'shot_{shot_number}_frame{frame_number}.txt'
    number_of_clips = len(clips) // FRAMES_PER_SHOT
    labels = pd.DataFrame(columns=['path', 'is_shot'])
    for i in range(number_of_clips):
        shot_number = format_number(i)
        for j in range(FRAMES_PER_SHOT):
            frame = picture_string.format(shot_number=shot_number, frame_number=format_number(j))
            is_shot = 0
            if frame in shots:
                is_shot = 1
            labels.loc[len(labels)] = [frame, is_shot]
    labels.to_csv('labels.csv', index=False)

if __name__ == "__main__":
    load_data()
