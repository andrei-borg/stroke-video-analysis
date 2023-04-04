# Code to split videos to frames

import cv2

# Object of class VideoCapture, obtain frames from a video. INSERT PATH TO VIDEO
capture = cv2.VideoCapture('/Users/elsathorestrom/Documents/Chalmers/Kanditatarbete/Egen programmering/Facialispares 3 - Elsa - 4.mp4')

# Define variable that tracks the number of current frames we are processing, starts at zero
frameNr = 0

# Start reading frames in an infinite loop, break when there are no more frames
while (True):
    # Process frames	
    success, frame = capture.read()
    if success:
        cv2.imwrite(f'/Users/elsathorestrom/Documents/Chalmers/Kanditatarbete/Egen programmering/frames_facialispares_3/frame_{frameNr}.jpg', frame)
    else:
        break
    frameNr = frameNr+1

capture.release()