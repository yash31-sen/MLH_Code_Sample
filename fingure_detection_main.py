#importing modules:
import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# aditional modules for functions
from screen_brightness_control import get_brightness, set_brightness

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands, draw=True, display = True):

    op_img = image.copy()
    
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    res = hands.process(imgRGB)
    

    if res.multi_hand_landmarks and draw:
    
        for hand_landmarks in res.multi_hand_landmarks:
          
            mp_drawing.draw_landmarks(image = op_img, landmark_list = hand_landmarks,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                     thickness=2, circle_radius=2))
    if display:
        
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(op_img[:,:,::-1]);plt.title("Output");plt.axis('off');

    else:

        return op_img, res

image = cv2.imread('media/sample.jpg')
detectHandsLandmarks(image, hands, display=True)


def countFingers(image, res, draw=True, display=True):

    height, width, _ = image.shape
    
    op_img = image.copy()
    
    count = {'RIGHT': 0, 'LEFT': 0}
    
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
  
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    
    for hand_index, hand_info in enumerate(res.multi_handedness):
        
        hand_label = hand_info.classification[0].label
        
        hand_landmarks =  res.multi_hand_landmarks[hand_index]
      
        for tip_index in fingers_tips_ids:
            
            finger_name = tip_index.name.split("_")[0]
          
            if (hand_landmarks.landmark[tip_index].y &lt; hand_landmarks.landmark[tip_index - 2].y):
                
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
              
                count[hand_label.upper()] += 1

        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
 
        if (hand_label=='Right' and (thumb_tip_x &lt; thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x &gt; thumb_mcp_x)):
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            count[hand_label.upper()] += 1

    if draw:

        cv2.putText(op_img, " Total Fingers: ", (10, 25),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)
        cv2.putText(op_img, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20,255,155), 10, 10)

    if display:
       
        plt.figure(figsize=[10,10])
        plt.imshow(op_img[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:

        return op_img, fingers_statuses, count

camera_video = cv2.VideoCapture(1)
camera_video.set(3,1280)
camera_video.set(4,960)

cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)

while camera_video.isOpened():
    
    ok, frame = camera_video.read()

    if not ok:
        continue
     
    frame = cv2.flip(frame, 1)
  
    frame, res = detectHandsLandmarks(frame, hands_videos, display=False)

    if res.multi_hand_landmarks:
    
        frame, fingers_statuses, count = countFingers(frame, res, display=False)              
  
    cv2.imshow('Fingers Counter', frame)
    
    k = cv2.waitKey(1) &amp; 0xFF
    
    # esc
    if(k == 27):
        break

camera_video.release()
cv2.destroyAllWindows()



# -----------------------------------------------------------------------------
def recognizeGestures(image, fingers_statuses, count, draw=True, display=True):
 
    op_img = image.copy()
    
    hands_labels = ['RIGHT', 'LEFT']
    
    hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}
    
    for hand_index, hand_label in enumerate(hands_labels):

#screenshot
       
        elif count[hand_label] == 1:
            screenshot = pyautogui.screenshot()
            screenshot.save("screenshot.png")
# brightness increase

        elif count[hand_label] == 2:
            current_brightness = get_brightness()
            new_brightness = current_brightness+5
            set_brightness(new_brightness)
# brightness decrease

        elif count[hand_label] == 3:
            current_brightness = get_brightness()
            new_brightness = current_brightness-5
            set_brightness(new_brightness)
# volume up
        elif count[hand_label] == 4:
            pyautogui.press('volumeup')
# volume down

        elif count[hand_label] == 5:
            pyautogui.press('volumedown')
      
        if draw:
            # Write the hand gesture on the output image. 
            cv2.putText(op_img, hand_label +': '+ hands_gestures[hand_label] , (10, (hand_index+1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, color, 5)
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(op_img[:,:,::-1]);plt.title("Output Image");plt.axis('off')
    
    # Otherwise
    else:

       return op_img, hands_gestures