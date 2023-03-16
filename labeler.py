import cv2
import os
import matplotlib.pyplot as plt
import keyboard
import csv

def labeler(video_path,label_path):
    label = []
    video = cv2.VideoCapture(video_path)
    vid, frame = video.read()
    
    k = 0
    try:
        while(vid):
            i = 0
            vid, frame = video.read()
            if vid:
                cv2.imshow("frame1",frame)
                cv2.waitKey(0)
                i = keyboard.read_key()
                print(i)
                label.append(i)
            else:
                break
    except KeyboardInterrupt:
        pass
    f = open(label_path, 'w')

    writer = csv.writer(f)
    writer.writerow(label)
    f.close()
    return label


labeler(r"C:\Users\oskar\Documents\repo\stroke-video-analysis\C00030j.MP4",r"C:\Users\oskar\Documents\repo\stroke-video-analysis\label1.csv")


