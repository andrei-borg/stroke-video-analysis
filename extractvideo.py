
import cv2
import os
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import mediapipe as mp
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

def extract_face(data):
    print("Processing: "+str(data)+"...")
    video = cv2.VideoCapture(r'C:\Users\oskar\Documents\repo\stroke-video-analysis\test\\'+ data)
    vid, frame = video.read()
    faces = []
    curr_frame = 0
    path = data.replace(".mp4","")
    if os.path.exists(r"C:\Users\oskar\Documents\repo\stroke-video-analysis\faces\\" + path):
        return
    os.mkdir(r"C:\Users\oskar\Documents\repo\stroke-video-analysis\faces\\" + path)
    while(vid):
        curr_frame +=1
        vid, frame = video.read()
        if vid:
            face = media_pipe_detection(frame)
            
            name = r"C:\Users\oskar\Documents\repo\stroke-video-analysis\faces\\" + path + "\\" + path + 'frame' + str(curr_frame) + '.jpg'
            cv2.imwrite(name, face)
        else:
            return
    
    
    video.release()
    cv2.destroyAllWindows()
    
    print("Done Processing:" + str(curr_frame) + " faces found")
    
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

#extract_frames(r"C:\Users\oskar\Documents\repo\stroke-video-analysis\frames",r"C:\Users\oskar\Documents\repo\stroke-video-analysis\C0003.MP4")
#delete_frames("frames")
'''media pipe face detection'''
def media_pipe_detection(picture):

    mp_face_detection = mp.solutions.face_detection

    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    sample_img = picture
    
    height = len(sample_img[:,:,::-1])
    width = len(sample_img[:,:,::-1][0])

    face_detection_results = face_detection.process(sample_img[:,:,::-1])
    box = face_detection_results.detections[0].location_data.relative_bounding_box
    xmin = int(box.xmin * width)
    ymin = int(box.ymin * height)
    xmax = int(box.width * width + box.xmin * width)
    ymax = int(box.height * height + box.ymin * height)

    crop_img = sample_img[ymin:ymax,xmin:xmax]

    return crop_img[:,:,::]

def extract_face(data):
    print("Processing: "+str(data)+"...")
    video = cv2.VideoCapture(r'C:\Users\oskar\Documents\repo\stroke-video-analysis\test\\'+ data)
    vid, frame = video.read()
    faces = []
    curr_frame = 0
    path = data.replace(".mp4","")
    if os.path.exists(r"C:\Users\oskar\Documents\repo\stroke-video-analysis\faces\\" + path):
        return
    os.mkdir(r"C:\Users\oskar\Documents\repo\stroke-video-analysis\faces\\" + path)
    while(vid):
        curr_frame +=1
        vid, frame = video.read()
        if vid:
            face = media_pipe_detection(frame)
            
            name = r"C:\Users\oskar\Documents\repo\stroke-video-analysis\faces\\" + path + "\\" + path + 'frame' + str(curr_frame) + '.jpg'
            cv2.imwrite(name, face)
        else:
            return

    
    
    video.release()
    cv2.destroyAllWindows()
    
    print("Done Processing:" + str(curr_frame) + " faces found")

i = 0
for data in os.listdir(r'C:\Users\oskar\Documents\repo\stroke-video-analysis\test'):
    i += 1
    if i == 10:
        break
    extract_face(data)