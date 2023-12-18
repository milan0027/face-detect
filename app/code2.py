"""File contains Models and Functions for liveliness, blink and cover ratio detection using approach 1"""
#Imports
import cvzone
import numpy as np
import tensorflow as tf
import base64
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Union
import math

# Loading all the models

base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector_ = vision.FaceDetector.create_from_options(options)
mask_model = tf.keras.models.load_model('mask_model3.h5', compile=False)
model = tf.keras.models.load_model('liveliness_model.h5', compile=False)
classNames = ["Live","Spoof"]
detector = FaceDetector()

eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = 'haarcascade_frontalface_alt.xml'  #face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def face_landmarks(img):
    """
    Function to get the co-ordinates of eyes, nose, lips and chin
    """
    img = np.array(img)
    height, width, _ = img.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector_.detect(mp_image)
    for detection in detection_result.detections:
    # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        # Draw keypoints
        temp = 0
        face_lndm = []
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            face_lndm.append(keypoint_px)
            temp+=1
            if temp == 4:
               break
        if(temp == 4):
            return face_lndm
        
    return []
def face_mask(img):
    """
    Function to identify whether the mask is worn, not worn or worn incorrectly
    """
    results = mask_model.predict(img)
    results = results[0]
    #0->No mask, 1->Mask, 2->mask on chin
    return results

  
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

def convert_to_base64(img, quality=90):
    try:
        _, im_arr = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
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
    offsetPercentageH = 5
    offsetMask = 5
    img = convert_from_base64(encoded_string)
    if img is not None:
    #Detect faces in the image
        img2, bboxs = detector.findFaces(img,draw=False)
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
            xc = int(x - 2*offsetW)
            xw = int(w + 4*offsetW)
            offsetH = (offsetPercentageH/100)*h
            yp = int(y - offsetH * 6)
            hp = int(h + offsetH * 6)
            yc = int(y - offsetH * 7)
            yh = int(h + offsetH * 11)
            
            # Ensure that x, y, w, and h stay within image dimensions
            xp = max(0, xp)
            yp = max(0, yp)
            wp = min(img.shape[1] - xp, wp)
            hp = min(img.shape[0] - yp, hp)
            
            xc = max(0, xc)
            yc = max(0, yc)
            xw = min(img.shape[1] - xc, xw)
            yh = min(img.shape[0] - yc, yh)
            cropped_face = img[yc:yc+yh, xc:xc+xw]
            
            
            # Resize the raw image into (224-height,224-width) pixels
            image = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
            # Make the image a numpy array and reshape it to the models input shape.
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            # Normalize the image array
            image = (image / 127.5) - 1

            #Get the probabilites of the mask scenario
            prob_arr = face_mask(image)

            #Get the facial landmarks
            face_lndmk = face_landmarks(img)
            if(len(face_lndmk) == 0):
                cvzone.cornerRect(img, (xp, yp, wp, hp),colorC=(255, 0, 0),colorR=(255, 0, 0))
            
                return (convert_to_base64(img), 0, 0, 0)
            

            #Calculations for the upper limit of mask's bounding box
            left_eye_x, left_eye_y = face_lndmk[0]
            right_eye_x, right_eye_y = face_lndmk[1]
            nose_x, nose_y = face_lndmk[2]
            lips_x, lips_y = face_lndmk[3]

            with_mask_y = (left_eye_y+right_eye_y+(2*nose_y))/4
        
            incorrect_mask_y = lips_y
        
            mask_bbox_x2 = xp+wp
            without_mask_y = yp+hp
            
            mask_bbox_y1 = int(with_mask_y*prob_arr[1] + without_mask_y*prob_arr[0] + incorrect_mask_y*prob_arr[2])
            mask_bbox_y1 = int((1 + offsetMask/100)*mask_bbox_y1)
            
            mask_bbox_y2 = int(without_mask_y)
        
            mask_bbox_x1 = xp
            mask_bbox_w = wp
            mask_bbox_h = max(0, mask_bbox_y2-mask_bbox_y1)
            cover_ratio = mask_bbox_h / hp
        
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
            class_name = classNames[cls]
            confidence_score = prediction[0][cls]
            
            
            live_percentage = confidence_score
            if classNames[cls]=='Spoof':
                live_percentage = 1-confidence_score
            
            color = (0, 255, 0)
            
            cvzone.cornerRect(img, (xp, yp, wp, hp),colorC=color,colorR=color)

            return (convert_to_base64(img), blink, live_percentage, cover_ratio)

    
    return (encoded_string, 0, -1, -1)