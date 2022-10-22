import cv2
import mediapipe as mp

def check_camera(frame_name='Screen', device_id=0):
    '''
    Ensure that the correct camera is being selected or even that a particular camera is ON
    args:
    frame_name - Title that you want the window to display as the title of the screen
    device_id - This is the device id of the camera that you want to use. If there is only one camera attached to your system, by default its id is 0
    '''

    cap = cv2.VideoCapture(device_id) #If you have only one camera, then its device id is by default 0. Else different cameras attached to your system will have different ids
    while cap.isOpened():
        isframe, frame=cap.read()
        cv2.imshow(frame_name,frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def check_hand_landmarks(mp_draw_object, mp_hands_object, device_id=0, frame_name='Hands'):
    '''
    Check if mediapipe is accurately detecting our hands. Mediapipe's output tends to depend slightly on light and contrast between the hand color and the background.
    This function tracks and displays the landmarks of the hand before we can confirm that we want to start capturing the landmarks as data.
    args:
    frame_name - Title that you want the window to display as the title of the screen
    device_id - This is the device id of the camera that you want to use. If there is only one camera attached to your system, by default its id is 0
    '''
    mp_draw=mp.solutions.drawing_utils
    mp_hands=mp.solutions.hands
    cap=cv2.VideoCapture(device_id)
    i=0
    lm=[]
    with mp_hands_object.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1) as hands:
        while cap.isOpened():
            isframe, frame=cap.read()
            if not isframe:
                print('Empty Frame')
                continue
            else:
                results=hands.process(frame)
                if results.multi_hand_landmarks:
                    for landmark in results.multi_hand_landmarks:
                        if i==0:
                            lm.append(landmark)
                            i+=1   
                        mp_draw_object.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)
                cv2.imshow(frame_name,frame)
                if cv2.waitKey(10) & 0xFF==ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()
    return lm

# check_camera()
# lm1=check_hand_landmarks()
# print('Functions run')
# print(lm1)




                    