from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import csv
import cv2
import numpy as np
import os
from datetime import timedelta
import glob


def format_number(number):
    if number < 10:
        return f'00{number}'
    elif number < 100:
        return f'0{number}'
    else:
        return str(number)


def cut_video(video, textfile, length):
    times = []
    with open(textfile, 'r', encoding='utf-8-sig') as file:
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader:
            times.append(float(row[0]))

    for number, time in enumerate(times):
        minutes, seconds = divmod(time, 1)  # Separating minutes and seconds
        seconds = round((minutes * 60) + (seconds * 100))  # Converting to seconds
        start = seconds - (length / 2)
        end = seconds + (length / 2)
        ffmpeg_extract_subclip(video, start, end, targetname=f"cut/shot_{format_number(number)}.mp4")


def extract_frames(video, frames_per_second):
    count = 0
    name = video[4:-4]
    out = "frames/"
    vidcap = cv2.VideoCapture(video)
    print(out)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 75))
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        if success:
            cv2.imwrite(fr"C:\Users\fziet\OneDrive\Pulpit\szkola\inzynierka\frame_extracting\frames\{name}_frame{format_number(count)}.jpg", image)  # save frame as JPEG file
            count = count + 1
        else:
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #cut_video('videos/gole.mp4', 'text/mov1.csv', 2)
    for i in glob.glob("cut/*"):
        extract_frames(i, 30)

