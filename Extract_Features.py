import mediapipe as mp
import cv2
import os
import pickle

mp_hand=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_style=mp.solutions.drawing_styles
hand=mp_hand.Hands(static_image_mode=True,min_detection_confidence=0.5)
Data_dir = "./data"
data=[]
labels=[]
for dir in os.listdir(Data_dir):
    for img_path in os.listdir(os.path.join(Data_dir,dir)):
        data_aux=[]
        x_=[]
        y_=[]
        img=cv2.imread(os.path.join(Data_dir,dir,img_path))
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=hand.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for i in range(len(hand_landmark.landmark)):
                    x=hand_landmark.landmark[i].x
                    y=hand_landmark.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmark.landmark)):  #Normalisation
                    x=hand_landmark.landmark[i].x
                    y=hand_landmark.landmark[i].y
                    data_aux.append(x-min(x_))
                    data_aux.append(y-min(y_))

        data.append(data_aux)
        labels.append(dir)

with open("Data.picle",'wb') as file:
    pickle.dump({"data":data , "labels":labels},file)

