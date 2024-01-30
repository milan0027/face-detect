"""File contains Models and Functions for liveliness, blink and cover ratio detection using approach 2.
This approach gives better results on average than approach 1 and handles various complex scenarios"""
#Imports
import cvzone
import numpy as np
import tensorflow as tf
import base64
from ultralytics import YOLO
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Loading all the models

base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector_ = vision.FaceDetector.create_from_options(options)
model = tf.keras.models.load_model('liveliness_model.h5',compile=False)
classNames = ["Live","Spoof"]
detector = FaceDetector()
uncover_model = YOLO('new_masks_added.pt')
classNames_uncover = [
      "Uncovered",
      "Hand_and_Object",
      "Helmet_and_Cap",
      "Mask",
      "Scarf",
      "Spectacles",
]
eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = 'haarcascade_frontalface_alt.xml'  #face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)


def visualize(detection_result):
  
  
  for detection in detection_result.detections:

    category = detection.categories[0]
    probability = round(category.score, 4)
   
    if(probability >= 0.92):
        return 0
  
    if(probability <= 0.55):
        return 0.85
    
    ratio = 2.36 * ( 0.92 - probability)
    return ratio

  return 0.85


def coverratio(img):

    results = uncover_model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            
            
            # Class Name
            cls = int(box.cls[0])
            
            cover = -1
            if classNames_uncover[cls] == "Spectacles":
                cover = 0.2
            elif classNames_uncover[cls] == "Helmet_and_Cap":
                cover = 0.15
            elif classNames_uncover[cls] == "Scarf":
                cover = 0.1
            elif classNames_uncover[cls] == "Mask":
                cover = 0.45
            elif classNames_uncover[cls] == "Uncovered":
                cover = 0
            
            print(cover)
            return cover

        return 0.85
    
    return 0.85  
      
        


def detect_eyes(img):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        
        frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
        eyes = eyeCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        if len(eyes) == 0:
            return 2 # closed eyes
        else:
            return 3 # open eyes
    return 0 # no conclusion


def convert_from_base64(encoded_string):
    try:
        # Directly decode using cv2.imdecode and np.frombuffer
        im_arr = np.frombuffer(base64.b64decode(encoded_string), dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None  # Handle the error accordingly in your application

def convert_to_base64(img):
    try:
        _, im_arr = cv2.imencode('.jpg', img)
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        return im_b64.decode()
    except Exception as e:
        print(f"Error encoding to base64: {e}")
        return None  # Handle the error accordingly in your application




def combined(encoded_string): 
    """Main function"""

    #Offset percentages of all the bounding boxes

    offsetPercentageW = 5
    offsetPercentageH = 2.5
    img = convert_from_base64(encoded_string)
    
    if img is not None:
    #Detect faces in the image
        img2, bboxs = detector.findFaces(img, draw=False)
        # (image, multiple, helmet confidence, mask confidence, live confidence, cover ratio)

        # If more than 1 face, return the image itself
        if(len(bboxs) > 1):
        #print('here')
            return (encoded_string, 1, -1, -1)
        
        # Only proceed when there is only 1 face
        if(len(bboxs) == 1):
            bbox = bboxs[0]
            x,y,w,h = bbox["bbox"]
            
            # Set the offset for the face's bounding box
            offsetW = (offsetPercentageW/100)*w
            xp = int(x - offsetW)
            wp = int(w + 2*offsetW)
            offsetH = (offsetPercentageH/100)*h
            yp = int(y - offsetH * 6)
            hp = int(h + offsetH * 6)
            
            # Ensure that x, y, w, and h stay within image dimensions
            xp = max(0, xp)
            yp = max(0, yp)
            wp = min(img.shape[1] - xp, wp)
            hp = min(img.shape[0] - yp, hp)
            
            
            offsetPercentageW = 10
            offsetPercentageH = 15
            
            offsetW = (offsetPercentageW/100)*w
            x = int(x - offsetW)
            w = int(w + offsetW * 2)
            
            offsetH = (offsetPercentageH/100)*h
            y = int(y - offsetH * 3)
            h = int(h + offsetH * 4)
            
            # Ensure that x, y, w, and h stay within image dimensions
            x = max(0, x)
            y = max(0, y)
            w = min(img.shape[1] - x, w)
            h = min(img.shape[0] - y, h)
            
            # cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)
            
            cropped_face = img[y:y+h, x:x+w]
            vision_img = np.array(cropped_face)
            type(vision_img)
            rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=vision_img)
            detection_result = detector_.detect(rgb_image)
            cover_ratio = visualize(detection_result)
            cover_ratio2 = coverratio(img)
            if(cover_ratio2 != -1):
                cover_ratio = (cover_ratio + cover_ratio2)/2
            blink = detect_eyes(img)
                
            # Resize the raw image into (224-height,224-width) pixels
            image = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
            # Make the image a numpy array and reshape it to the models input shape.
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            # Normalize the image array
            image = (image / 127.5) - 1
            
            # Predicts the model
            prediction = model.predict(image)
            cls = np.argmax(prediction)
            confidence_score = prediction[0][cls]
            
            
            live_percentage = confidence_score
            if classNames[cls]=='Spoof':
                live_percentage = 1-confidence_score
            
            color = (0, 255, 0)
            
            cvzone.cornerRect(img, (xp, yp, wp, hp),colorC=color,colorR=color)

            live_percentage = round(live_percentage, 4)
            cover_ratio = round(cover_ratio, 4)
            return (convert_to_base64(img), blink, live_percentage, cover_ratio)

    
    return (encoded_string, 0, -1, -1)