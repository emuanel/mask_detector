# DATA: vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
import cv2
import dlib
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from pascal_voc_writer import Writer
# read data
mask_paths = glob.glob("masks\*.png")
face_paths = glob.glob("lfw-deepfunneled\*\*.jpg")

#number_photos = 1000
#face_paths = face_paths[:number_photos]

#Image generator 
def ReadImages(path):
    for i in path:
        yield cv2.imread(i)
images = ReadImages(face_paths)

#mask image
masks = []
for i in mask_paths:
    masks.append(cv2.imread(i))

# Initialization of the keypointow predictor
detector = dlib.get_frontal_face_detector()
p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)



def add_mask(img, landmarks, mask):
    interesting_landmarks = {4:(0,0), 8:(0,0), 12:(0,0), 29:(0,0)}                      #landmarks for add mask
    for key, value in interesting_landmarks.items():
        interesting_landmarks[key] = landmarks.part(key).x, landmarks.part(key).y       #read loc of landmarks
        # cv2.circle(img, interesting_landmarks[key], 4, (0, 0, 255), -1)
        
    top, bottom = interesting_landmarks.get(29)[1], interesting_landmarks.get(8)[1]     
    left, right = interesting_landmarks.get(4)[0], interesting_landmarks.get(12)[0]
    
    high = bottom - top
    lenght = right - left
    mask = cv2.resize(mask, (lenght, high))
    
    try:
        for row in range(top, bottom):
            for column in range(left, right):
                img[row, column] = mask[row - top, column - left] if (mask[row - top, column - left] != [0, 0, 0]).all() else img[row, column]
        return img  
    except:
        print("ERROR") 
        return img  

#path to save data
data_path = "data\\"

def save_pascalVOC_addnotations(image_path, path_to_save, label, left_top, right_bottom):
    writer = Writer(image_path, 250, 250)
    writer.addObject(label, left_top[0], left_top[1], right_bottom[0], right_bottom[1]) 
    writer.save(path_to_save)            
        
          
for i in range(len(face_paths)):
    img = next(images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    faces = detector(img)                               #face detection
    
    for face in faces:
        landmarks = predictor(img, face)                #landmarks detection
        left_top = face.left(), face.top()              #left top corner of face
        right_bottom = face.right(), face.bottom()      #left bottom corner of face
        # cv2.rectangle(img, left_top, right_bottom, (0,255,0), 3)
       
    image_path = data_path+'images\\{}.jpg'.format(i)
    annotations_path = data_path+'annotations\\{}.xml'.format(i)
    if(i%2):    
        img = add_mask(img, landmarks, masks[i%4])
        save_pascalVOC_addnotations(image_path, annotations_path, "mask", left_top, right_bottom)
        cv2.imwrite(image_path, img)
    else:
        save_pascalVOC_addnotations(image_path, annotations_path, "without_mask", left_top, right_bottom)
        cv2.imwrite(image_path, img)
        
       
 #   cv2.imshow(("Photo: {}".format(i)),img)
 #   cv2.waitKey()
        