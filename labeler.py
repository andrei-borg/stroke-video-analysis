import cv2
import os
import keyboard
import csv
import numpy as np
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


#labeler(r"C:\Users\oskar\OneDrive\Dokument\repo\kandidat\video\klipptavideor\Facialispares 3 - Oskar - 10.mp4",r"C:\Users\oskar\OneDrive\Dokument\repo\kandidat\video\labels\Facialispares 3 - oskar -  label.csv")


def auto_label(Video_from,csv_to):
    
    video = cv2.VideoCapture(Video_from)
    vid, frame = video.read()
    label = []
    while(vid):
        
        vid, frame = video.read()
        if vid:
            label.append(0) 
        else:
            break
    f = open(csv_to, 'w')
    writer = csv.writer(f)
    writer.writerow(label)
    f.close()
    print(label)
    return label

auto_label(r"C:\Users\oskar\OneDrive\Dokument\repo\kandidat\video\klipptavideor\Facialispares 0 - Oskar - 6.mp4",r"C:\Users\oskar\OneDrive\Dokument\repo\kandidat\video\labels\Facialispares 0 - Oskar - 6 label.csv")