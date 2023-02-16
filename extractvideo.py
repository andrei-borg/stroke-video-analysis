
import cv2
import os
from retinaface import RetinaFace
import matplotlib.pyplot as plt

def extract_frames(path_to,path_from):
    
    video = cv2.VideoCapture(path_from)
    
    curr_frame = 0
    os.chdir(path_to)
    while(True):
        
        vid, frame = video.read()
        
        if vid:
            name = path_to + str(curr_frame) + '.jpg'
            print('Creating: '+ name)
            
            cv2.imwrite(name, frame)
            
            curr_frame += 1
        else:
            break
    
    video.release()
    cv2.destroyAllWindows()
    
def delete_frames(path):
    os.chdir(path)
    for i in range(744):
        os.remove("frames"+str(i)+".jpg")
    
def obtain_face(img_path):
    faces = RetinaFace.extract_faces(img_path)
    for face in faces:
        plt.imshow(face)
        plt.show()
    return faces[0]
#extract_frames("frames",r"C:\Users\oskar\OneDrive\Dokument\repo\kandidat\video\1080 50fps - 2.MP4")
#delete_frames("frames")

