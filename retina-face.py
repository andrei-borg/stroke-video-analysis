from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Nathalie.jpg")

obj = RetinaFace.detect_faces("Nathalie.jpg")

for key in obj.keys():
    identity = obj[key]

    facial_area = identity['facial_area']

    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 2)

plt.figure(figsize=(8, 8))
ax1.axis('off')
plt.imshow(img[:, :, ::-1])
plt.show()

