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
