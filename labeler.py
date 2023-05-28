import cv2
import os
import keyboard
import csv


# Code for creating labels in video frames
def labeler(video_path, label_path):
    label = []
    video = cv2.VideoCapture(video_path)
    vid, frame = video.read()

    k = 0
    try:
        while vid:
            i = 0
            vid, frame = video.read()
            if vid:
                cv2.imshow("frame1", frame)
                cv2.waitKey(0)
                i = keyboard.read_key()
                print(i)
                label.append(i)
            else:
                break
    except KeyboardInterrupt:
        pass
    f = open(label_path, "w")

    writer = csv.writer(f)
    writer.writerow(label)
    f.close()
    return label
