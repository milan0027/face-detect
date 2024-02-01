"""File contains Models and Functions for liveliness, blink and cover ratio detection using approach 2.
This approach gives better results on average than approach 1 and handles various complex scenarios"""
#Imports
import cvzone
import numpy as np
import base64
from ultralytics import YOLO
import cv2
import math
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from shapely.geometry import Polygon
# Loading all the models
detector = FaceDetector()
# Initialize the HandDetector class with the given parameters
detector_hand = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
model = YOLO('new_masks_added.pt')
classNames = [
      "Uncovered",
      "Hand_and_Object",
      "Helmet_and_Cap",
      "Mask",
      "Scarf",
      "Spectacles",
]


def cover_result(img):

    results = model(img, stream=True, verbose=False)
    hands, img = detector_hand.findHands(img, draw=False, flipType=True)
    
    
    for r in results:
        boxes = r.boxes
        
        x1, y1, x2, y2 = 0,0,0,0

    
        box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygon_face = Polygon(box_face)
        total_face_area = 1

        uncover = 1
        clsname = "Covered"
        
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            conf1 = conf
            # Class Name
            cls = int(box.cls[0])
            
            box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            polygon_face = Polygon(box_face)
            total_face_area = polygon_face.area
              
            uncover = 1
            clsname = "Covered"
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet_and_Cap":
              clsname = "Uncovered"
              uncover = 0.85 
            elif classNames[cls] == "Scarf":
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              uncover = 0.6 
             
            

            
  
        # Check if any hands are detected
        for hand in hands:
                
            # Information for the first hand detected
            hand1 = hand  # Get the first hand detected
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")
                
                    
            (x1, y1, w, h) = bbox1
            x2 = x1 + w
            y2 = y1 + h
                
            box_hand1 = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                
            polygon_hands = Polygon(box_hand1)
            polygon_face = polygon_face - polygon_face.intersection(polygon_hands)


            uncover = uncover * ( polygon_face.area / total_face_area )
            if uncover < 0.95:
              clsname = "Covered"
            
        
        
        
        covered = (clsname=="Covered")

        return covered
    
    return 1
      
def draw_result(img):
    results = model(img, stream=True, verbose=False)
    hands, img = detector_hand.findHands(img, draw=False, flipType=True)
    
    
    for r in results:
        boxes = r.boxes
        #print("Faces detected: ",len(boxes))
        # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
        
        x1, y1, x2, y2 = 0,0,0,0
         
        xf = x1
        yf = y1
        wf = 0
        hf = 0
        color = (0, 255, 0)
        conf1 = 0

    
        box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygon_face = Polygon(box_face)
        total_face_area = 1

        uncover = 1
        clsname = "Covered"
        
        if(len(boxes)>1):
          img, bboxs = detector.findFaces(img,draw=True)
          return img

        if(len(boxes)==0):
          img, bboxs = detector.findFaces(img,draw=False)
          
          for bbox in bboxs:
            x,y,w,h = bbox["bbox"]
                      
            offsetPercentageW = 10
            offsetPercentageH = 20
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
            
            x1 = x
            y1 = y
            x2 = x1 + w
            y2 = y1 + h
            
            xf = x1
            yf = y1
            
            uncover = 0.7
            
            box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            polygon_face = Polygon(box_face)
            total_face_area = polygon_face.area
            
            cvzone.cornerRect(img, (x,y,w,h), (255,0,0), 3)
            
            cvzone.cornerRect(img, (x, y, w, h),colorC=(0,0,255),colorR=(0,0,255))
        
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            conf1 = conf
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            
            xf = x1
            yf = y1
            wf = w
            hf = h
            
            box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            polygon_face = Polygon(box_face)
            total_face_area = polygon_face.area
              
            uncover = 1
            clsname = "Covered"
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet_and_Cap":
              clsname = "Uncovered"
              uncover = 0.85 
            elif classNames[cls] == "Scarf":
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              uncover = 0.6 
             
            if clsname == "Covered":
              color=(0,0,255)
            
  
        # Check if any hands are detected
        for hand in hands:
                
            # Information for the first hand detected
            hand1 = hand  # Get the first hand detected
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")
                
                    
            (x1, y1, w, h) = bbox1
            x2 = x1 + w
            y2 = y1 + h
                
            box_hand1 = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                
            polygon_hands = Polygon(box_hand1)
            polygon_face = polygon_face - polygon_face.intersection(polygon_hands)


            uncover = uncover * ( polygon_face.area / total_face_area )
            if uncover < 0.95:
              clsname = "Covered"
            
        if clsname == "Covered":
          color=(0,0,255)
        
        cvzone.cornerRect(img, (xf, yf, wf, hf),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{clsname}',
                               (max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,
                               colorB=color)
        
        return img
    
    return img

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
           
            cover_ratio = cover_result(img)
            live_percentage = -1
            cover_ratio = round(cover_ratio, 4)
            return (convert_to_base64(img), 0, live_percentage, cover_ratio)

    
    return (encoded_string, 0, -1, -1)

def realtime(encoded_string):
    
    img = convert_from_base64(encoded_string)

    if img is not None:
       
       img = draw_result(img)
       return convert_to_base64(img)
    
    return encoded_string
    