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

def auto_label(video_from,csv_to,labeling):
    label = []
    video = cv2.VideoCapture(video_from)
    vid, frame = video.read()
    label = []
    while(vid):
    
        vid, frame = video.read()
        if vid:
            label.append(labeling)
        else:
            break
    f = open(csv_to, 'w')
    writer = csv.writer(f)
    writer.writerow(label)
    f.close()
    print(label)
    return label

<<<<<<< HEAD
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 1.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 1 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 2.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 2 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 3.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 3 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 4.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 4 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 5.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 5 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 6.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 6 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 7.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 7 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 8.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 8 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 9.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 9 label.csv")
labeler(r"facialparesvideo\Facialispares 3 - Andrei - 10.mp4",r"facialpareslabel\Facialispares 3 - Andrei - 10 label.csv")
#auto_label(r"facialparesvideo\Facialispares 0 - Viktor - 5.mp4",r"facialpareslabel\Facialispares 0 - Viktor - 5 label.csv",0)
=======
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
>>>>>>> 398fa370911e24cceb84cd0ec91cf02014d6444e
