import cv2
import mediapipe as mp
import numpy as np
import os

def save_frame(path,mark):
    if mark:
        left_array=np.array([[lm.x,lm.y,lm.z] for lm in mark[0].landmark]).flatten()
        right_array=np.array([[lm.x,lm.y,lm.z] for lm in mark[1].landmark]).flatten() if len(mark)>1 else np.zeros((21,3)).flatten()
    else:
        left_array=np.zeros((21,3)).flatten()
        right_array=np.zeros((21,3)).flatten()
    stack=np.hstack((left_array,right_array))
    np.save(path,stack)

def get_grid(path,frame,mp_object,save=True):
    marks=[]
    #Get all the landmarks for the frame
    frame.flags.writeable = False 
    grid=mp_object.process(frame)
    frame.flags.writeable = True
    if grid.multi_hand_landmarks:
        if save:
            save_frame(path,grid.multi_hand_landmarks)
        return grid.multi_hand_landmarks
    else:
        if save:
            save_frame(path,None)
        return None
        
def process_image(path, frame,mp_object,draw_object,hand_object):
    no_landmarks=0
    marks=get_grid(path,frame,mp_object)
    if marks:
        for mark in marks:
            draw_object.draw_landmarks(frame, mark,hand_object.HAND_CONNECTIONS)
    else:
        no_landmarks=1
    return frame, no_landmarks