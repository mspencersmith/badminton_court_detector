import cv2
import os.path
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

video = "path/to_video.mp4"
video_images = os.path.join(BASE_DIR, "path/to_image/")

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    exit(0)

frameFrequency = 1
total_frame = 0
id = 2873
while True:
    if id > 3074:
        break
    ret, frame = cap.read()
    if ret is False:
        break
    total_frame += 1
    if total_frame%frameFrequency == 0:
        id += 1
        image_name = f'{video_images}{str(id)}.jpg'
        cv2.imwrite(image_name, frame)
        print(image_name)
cap.release()