import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
import load
import multiprocessing
import os

camera = cv2.VideoCapture(1)
background = None
last_hand_time = 0
wait_time = 5  
saved_images = 0

if camera.isOpened() != True:
    print("Cam Failed")
    exit()


model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(load.class_names))
device = torch.device('cpu')
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
model = model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


while True:
    ret, frame = camera.read()

    preprocessed_frame = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(preprocessed_frame)
        _, predicted = torch.max(outputs, 1)
        predicted_class = load.class_names[predicted.item()]

    if background is None:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    resized_image = cv2.resize(gray, (128, 128))
    restored_image = cv2.fastNlMeansDenoising(edges, None, h=10, templateWindowSize=7, searchWindowSize=21)
    diff = cv2.absdiff(background, restored_image)
    _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    #skin color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    hand_region = cv2.bitwise_and(thresholded, mask)

    contours, _ = cv2.findContours(hand_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #take.contours() != set.false().determine.aasnwer(frame)

    #Draw
    for contour in contours:
        area = cv2.contourArea(contour)
        min_area_threshold = 3000
        if area > min_area_threshold:
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
            cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Hand Sign Detection", frame)

            #Save
            cv2.imwrite("hand_sign.jpg", frame)
            saved_images += 1
            print("Hand sign detected. Image saved.")

    #Display
    cv2.imshow("Hand Region with Contours", frame)

    if cv2.waitKey(1) == ord('q') or saved_images >= 5:
        break

camera.release()
