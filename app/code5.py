"""File contains Models and Functions for multiple faces and cover ratio detection"""
#Imports
import cvzone
import numpy as np
import base64
from ultralytics import YOLO
import cv2
import math
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from shapely.geometry import Polygon, Point
import torch
from torchvision.ops import nms
# Initialize FaceDetector
detector = FaceDetector()
# Initialize the HandDetector class with the given parameters
detector_hand = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
#Loading the model
model = YOLO('model_2.pt')
classNames = [
      "Uncovered", 
      "Hand_and_Object",
      "Helmet",
      "Turban",
      "Cap",
      "Mask",
      "Scarf",
      "Spectacles",
  ]

def cover_result(img, bboxs):

    results = model(img, stream=True, verbose=False, iou=0.5)
    hands, img = detector_hand.findHands(img, draw=False, flipType=True)
    
    clsname = "Uncovered"
    x1, y1, x2, y2 = 0,0,0,0

    box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    polygon_face = Polygon(box_face)

    for r in results:
        boxes = r.boxes

        uncover = 1


        boxes_nms = torch.tensor([[int(box.xyxy[0][i]) for i in range(4)] for box in boxes])
        boxes_nms = boxes_nms.reshape(-1, 4)
        confs_nms = torch.tensor([math.ceil((box.conf[0] * 100)) / 100 for box in boxes])
        boxes_nms = boxes_nms.float()
        confs_nms = confs_nms.float()

        indices = nms(boxes = boxes_nms, scores = confs_nms, iou_threshold=0.2)
        
        for index in indices:
            box = boxes[index]
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Class Name
            cls = int(box.cls[0])
            
              
            uncover = 1
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet":
              uncover = 0.85 
            elif classNames[cls] == "Turban":
              clsname = "Uncovered"
            elif classNames[cls] == "Cap":
              clsname = "Uncovered"
            elif classNames[cls] == "Scarf":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              clsname = "Covered"
              uncover = 0.6 
            elif classNames[cls] == "Hand_and_Object":
              clsname = "Uncovered"
        
        break
             
            

    for bbox in bboxs:
        x,y,w,h = bbox["bbox"]
        score = int(bbox['score'][0] * 100)
        if(score < 90):
            clsname = "Covered"
        polygon_face = Polygon([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])  

    # Check if any hands are detected
    for hand in hands:
        # Information for the first hand detected
        hand1 = hand  # Get the first hand detected
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        
        for point in lmList1:
            [x,y,w]=point
            x,y,w=int(x),int(y),int(w)
            p = Point(x,y)
            if(polygon_face.contains(p)):
                clsname="Covered"
                break
        
    
    
    
    covered = (clsname=="Covered")

    return covered
      
def draw_result(img):
    img2, bboxs = detector.findFaces(img, draw=False)

    if(len(bboxs) != 1):
       return img
    
    clsname = "Uncovered"
    results = model(img, stream=True, verbose=False)
    hands, img = detector_hand.findHands(img, draw=False, flipType=True)
    
    x1, y1, x2, y2 = 0,0,0,0
         
    xf = x1
    yf = y1
    wf = 0
    hf = 0
    color = (0, 255, 0)
    #conf1 = 0


    box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    polygon_face = Polygon(box_face)

    for r in results:
        boxes = r.boxes
        
        
        #total_face_area = 1


        boxes_nms = torch.tensor([[int(box.xyxy[0][i]) for i in range(4)] for box in boxes])
        boxes_nms = boxes_nms.reshape(-1, 4)
        confs_nms = torch.tensor([math.ceil((box.conf[0] * 100)) / 100 for box in boxes])
        boxes_nms = boxes_nms.float()
        confs_nms = confs_nms.float()

        indices = nms(boxes = boxes_nms, scores = confs_nms, iou_threshold=0.2)
        uncover = 1
      
        for index in indices:
            box = boxes[index]
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            #conf = math.ceil((box.conf[0] * 100)) / 100
            #conf1 = conf
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            #xf,yf,wf,hf = x1,y1,w,h
            
              
            uncover = 1
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet":
              uncover = 0.85 
            elif classNames[cls] == "Turban":
              clsname = "Uncovered"
            elif classNames[cls] == "Cap":
              clsname = "Uncovered"
            elif classNames[cls] == "Scarf":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              clsname = "Covered"
              uncover = 0.6 
            elif classNames[cls] == "Hand_and_Object":
              clsname = "Uncovered"

        break   

    for bbox in bboxs:
        x,y,w,h = bbox["bbox"]
        xf,yf,wf,hf = x,y,w,h
        score = int(bbox['score'][0] * 100)
        if(score < 90):
            clsname = "Covered"
        polygon_face = Polygon([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])  
  
    # Check if any hands are detected
    for hand in hands:
        # Information for the first hand detected
        hand1 = hand  # Get the first hand detected
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        
        for point in lmList1:
            [x,y,w]=point
            x,y,w=int(x),int(y),int(w)
            p = Point(x,y)
            if(polygon_face.contains(p)):
                clsname="Covered"
                break
        
    if clsname == "Covered":
        color=(0,0,255)
    
    if(xf != 0 and yf != 0):
        cvzone.cornerRect(img, (xf, yf, wf, hf),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{clsname}',(max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,colorB=color)
    
    return img

def mixed_result(img, bboxs):

    results = model(img, stream=True, verbose=False, iou=0.5)
    hands, img = detector_hand.findHands(img, draw=False, flipType=True)
    
    clsname = "Uncovered"
    x1, y1, x2, y2 = 0,0,0,0
    xf = x1
    yf = y1
    wf = 0
    hf = 0
    color = (0, 255, 0)

    box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    polygon_face = Polygon(box_face)

    for r in results:
        boxes = r.boxes

        uncover = 1


        boxes_nms = torch.tensor([[int(box.xyxy[0][i]) for i in range(4)] for box in boxes])
        boxes_nms = boxes_nms.reshape(-1, 4)
        confs_nms = torch.tensor([math.ceil((box.conf[0] * 100)) / 100 for box in boxes])
        boxes_nms = boxes_nms.float()
        confs_nms = confs_nms.float()

        indices = nms(boxes = boxes_nms, scores = confs_nms, iou_threshold=0.2)
        
        for index in indices:
            box = boxes[index]
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Class Name
            cls = int(box.cls[0])
            
              
            uncover = 1
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet":
              uncover = 0.85 
            elif classNames[cls] == "Turban":
              clsname = "Uncovered"
            elif classNames[cls] == "Cap":
              clsname = "Uncovered"
            elif classNames[cls] == "Scarf":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              clsname = "Covered"
              uncover = 0.6 
            elif classNames[cls] == "Hand_and_Object":
              clsname = "Uncovered"
        
        break
             
            

    for bbox in bboxs:
        x,y,w,h = bbox["bbox"]
        xf,yf,wf,hf = x,y,w,h
        score = int(bbox['score'][0] * 100)
        if(score < 90):
            clsname = "Covered"
        polygon_face = Polygon([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])  

    # Check if any hands are detected
    for hand in hands:
        # Information for the first hand detected
        hand1 = hand  # Get the first hand detected
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        
        for point in lmList1:
            [x,y,w]=point
            x,y,w=int(x),int(y),int(w)
            p = Point(x,y)
            if(polygon_face.contains(p)):
                clsname="Covered"
                break
        
    
    if clsname == "Covered":
        color=(0,0,255)
    
    if(xf != 0 and yf != 0):
        cvzone.cornerRect(img, (xf, yf, wf, hf),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{clsname}',(max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,colorB=color)
    
    covered = (clsname=="Covered")

    return (img,covered)


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
        #(multiple_face, cover_ratio)
        # If more than 1 face
        if(len(bboxs) > 1):
            return (1, -1)
        
        # Only proceed when there is only 1 face
        if(len(bboxs) == 1):
           
            cover_ratio = cover_result(img, bboxs)
            cover_ratio = round(cover_ratio, 4)
            return ( 0, cover_ratio)

    
    return (0, -1)

def realtime(encoded_string):
    
    img = convert_from_base64(encoded_string)

    if img is not None:
       
       img = draw_result(img)
       return convert_to_base64(img)
    
    return encoded_string

def mixed(encoded_string):
    
    img = convert_from_base64(encoded_string)
    
    if img is not None:
    #Detect faces in the image
        img2, bboxs = detector.findFaces(img, draw=False)
        #(multiple_face, cover_ratio)
        # If more than 1 face
        if(len(bboxs) > 1):
            return (encoded_string, 1, -1)
        
        # Only proceed when there is only 1 face
        if(len(bboxs) == 1):
           
            img,cover_ratio = mixed_result(img, bboxs)
            cover_ratio = round(cover_ratio, 4)
            return (convert_to_base64(img), 0, cover_ratio)

    
    return (encoded_string, 0, -1)