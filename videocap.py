#IMPORTING REQUIRED LIBRARIES
from detecto.core import DataLoader, Model
from detecto import core, utils, visualize
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from detecto.utils import normalize_transform
import torch
import torchvision.transforms
from detecto.utils import read_image
import cv2
import numpy as np
import pandas as pd
from detecto.visualize import detect_live
import argparse

'''
Load the pretrained model-
Please change the path to the test images and Model below after downloading the Zip file.
Ensure the path is correct.

'''

model = Model.load(r'C:\Users\Prajwal V Athreya\Documents\Projects\facemask_detection\detector.pth',
                    ['with_mask', 'mask_weared_incorrect', 'without_mask'])

parser = argparse.ArgumentParser(description='For Thresholding Operations using inRange.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv2.VideoCapture(args.camera)

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    labels, boxes, scores = model.predict(frame)

    img = frame

    #parameters for bounding boxes
    thickness = 2 #pixels

    #convert the the tensors to a numpy array
    score = scores.numpy()
    boxes = boxes.numpy()

    min_score = 0.80   #to avoid drawing bounding boxes for predictions with low confidence
    i = 0

    #looping through the list of detections
    for score in scores:
        if score > min_score:
            if (labels[i] == 'with_mask'): #green bounding for people wearing a mask
                frame = cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0,255,0), thickness)
            elif (labels[i] == 'mask_weared_incorrect'): #blue bounding box for people weaing it incorrectly
                frame = cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0,0,255), thickness)
            elif (labels[i] == 'without_mask'): #red bounding box for people not wearing it
                frame = cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (255,0,0), thickness)
        i += 1

    frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)

    #Output the images
    #cv2.imshow('Original', frame)
    cv2.imshow('Final', frame)
    key = cv2.waitKey(30)
    if( key == ord('q') or key == 27):
        break

cv2.destroyAllWindows()
