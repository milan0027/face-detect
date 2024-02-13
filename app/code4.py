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
from shapely.geometry import Polygon
import torch
from torchvision.ops import nms
# Initialize FaceDetector
detector = FaceDetector()
# Initialize the HandDetector class with the given parameters
detector_hand = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
#Loading the model
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

    results = model(img, stream=True, verbose=False, iou=0.5)
    hands, img = detector_hand.findHands(img, draw=False, flipType=True)
    
    
    for r in results:
        boxes = r.boxes
        
        x1, y1, x2, y2 = 0,0,0,0

    
        box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygon_face = Polygon(box_face)
        total_face_area = 1

        uncover = 1
        clsname = "Covered"


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
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
                    
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
    img2, bboxs = detector.findFaces(img, draw=False)

    if(len(bboxs) != 1):
       return img
    
    
    results = model(img, stream=True, verbose=False)
    hands, img = detector_hand.findHands(img, draw=False, flipType=True)
    
    
    for r in results:
        boxes = r.boxes
        
        x1, y1, x2, y2 = 0,0,0,0
         
        xf = x1
        yf = y1
        wf = 0
        hf = 0
        color = (0, 255, 0)
        #conf1 = 0

    
        box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygon_face = Polygon(box_face)
        total_face_area = 1


        boxes_nms = torch.tensor([[int(box.xyxy[0][i]) for i in range(4)] for box in boxes])
        boxes_nms = boxes_nms.reshape(-1, 4)
        confs_nms = torch.tensor([math.ceil((box.conf[0] * 100)) / 100 for box in boxes])
        boxes_nms = boxes_nms.float()
        confs_nms = confs_nms.float()

        indices = nms(boxes = boxes_nms, scores = confs_nms, iou_threshold=0.2)
        uncover = 1
        clsname = "Covered"
      
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
            
            xf,yf,wf,hf = x1,y1,w,h
            
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
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
                    
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
        cvzone.putTextRect(img, f'{clsname}',(max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,colorB=color)
        
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
        #(multiple_face, cover_ratio)
        # If more than 1 face
        if(len(bboxs) > 1):
            return (1, -1)
        
        # Only proceed when there is only 1 face
        if(len(bboxs) == 1):
           
            cover_ratio = cover_result(img)
            cover_ratio = round(cover_ratio, 4)
            return ( 0, cover_ratio)

    
    return (0, -1)

def realtime(encoded_string):
    
    img = convert_from_base64(encoded_string)

    if img is not None:
       
       img = draw_result(img)
       return convert_to_base64(img)
    
    return encoded_string
    