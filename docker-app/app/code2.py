from ultralytics import YOLO
import cvzone
import numpy as np
import tensorflow as tf
import base64
from cvzone.FaceDetectionModule import FaceDetector
import cv2

model = tf.keras.models.load_model('keras_model2.h5',compile=False)
classNames = ["Live","Spoof"]
detector = FaceDetector()

def use_keras_after_zoom(encoded_string):

    im_bytes = base64.b64decode(encoded_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    img, bboxs = detector.findFaces(img,draw=False)
    
    if bboxs:
      # bboxInfo - "id","bbox","score","center"
      
      offsetPercentageW = 10
      offsetPercentageH = 20
      
      for bbox in bboxs:
        x,y,w,h = bbox["bbox"]
        # print(x,y,w,h)
        
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
        
        color = (0, 255, 0)
          
        # print(box.cls)
        if confidence_score > 0.6:

          if classNames[cls] == 'Live':
              color = (0, 255, 0)
              #print(f"Live Confidence = {confidence_score}, Class = {cls}")
          else:
              color = (0, 0, 255)
              #print(f"Spoof Confidence = {confidence_score}, Class = {cls}")

        cvzone.cornerRect(img, (x, y, w, h),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(confidence_score*100)}%',
                                   (max(0, x), max(35, y)), scale=2, thickness=4,colorR=color,
                                   colorB=color)
        
    _, im_arr = cv2.imencode('.jpg', img)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    

def demo(encoded_string):

    im_bytes = base64.b64decode(encoded_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
            
    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image array
    image = (image / 127.5) - 1
        
    # Predicts the model
    prediction = model.predict(image)
    cls = np.argmax(prediction)
    class_name = classNames[cls]
    confidence_score = prediction[0][cls]
        
    print("Clas: ",class_name," Conf: ",confidence_score)
        
    _, im_arr = cv2.imencode('.jpg', img)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    

    return im_b64.decode()
