
import cv2
import os
from retinaface import RetinaFace
import matplotlib.pyplot as plt

def extract_frames(path_to,path_from):
    
    video = cv2.VideoCapture(path_from)
    vid, frame = video.read()
    curr_frame = 0
    
    while(vid):
        
        vid, frame = video.read()
        if vid:
            name = path_to + '\\frame' + str(curr_frame) + '.jpg'
            print('Creating: '+ name)
            cv2.imwrite(name, frame)
            
            curr_frame += 1
        else:
            break
    
    video.release()
    cv2.destroyAllWindows()
    
def delete_frames(path):
    os.chdir(path)
    for i in range(2600):
        try:
            os.remove("frame"+str(i)+".jpg")
        except:
            pass
def obtain_face(img_path):
    faces = RetinaFace.extract_faces(img_path)
    for face in faces:
        plt.imshow(face)
        plt.show()
    return faces[0]
delete_frames(r"C:\Users\oskar\OneDrive\Dokument\repo\kandidat\frames")
#extract_frames(r"C:\Users\oskar\OneDrive\Dokument\repo\kandidat\frames",r"C:\Users\oskar\OneDrive\Dokument\repo\kandidat\video\108050fps-1.MP4")
#delete_frames("frames")

