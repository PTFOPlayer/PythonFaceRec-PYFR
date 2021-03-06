
from time import sleep
import cv2
import mediapipe as mp
import concurrent.futures
cap = cv2.VideoCapture(1)

draw = mp.solutions.drawing_utils

hand_model = mp.solutions.hands
hand_m = hand_model.Hands(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

face_model = mp.solutions.face_mesh
face_m = face_model.FaceMesh(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

pose_model = mp.solutions.pose
pose_m = pose_model.Pose(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
                         

class style(draw.DrawingSpec):
    def __init__(self, thickness=1, color=(0, 255, 0), circle_radius=1):
        self.thickness = thickness
        self.color = color
        self.circle_radius = circle_radius


def hands(frame, frgb):
    # hand detector
      
    results_h = hand_m.process(frgb)
    result = results_h.multi_hand_landmarks
    if result:
        for i, hand in enumerate(result):
            draw.draw_landmarks(frame, hand, hand_model.HAND_CONNECTIONS,
                                landmark_drawing_spec=style())
            
def faces(frame, frgb):
    # face detector
    results_f = face_m.process(frgb)
    result = results_f.multi_face_landmarks
    if result:
        for i, face in enumerate(result):
            draw.draw_landmarks(frame, face, 
                                landmark_drawing_spec=style())
            
def pose(frame, frgb):
    # pose detector
    results_p = pose_m.process(frgb)
    result = results_p.pose_landmarks
    if result:
        #for i, pose in enumerate(result):
            draw.draw_landmarks(frame, result, pose_model.POSE_CONNECTIONS,
                                landmark_drawing_spec=style())

def main():
    while True:
        _ , frame = cap.read()
        frame = cv2.flip(frame, 1)
        frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#        hands(frame, frgb)
#        faces(frame, frgb)
#        pose(frame, frgb)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            #executor.submit(hands, frame, frgb)
            executor.submit(faces, frame, frgb)
            executor.submit(pose, frame, frgb)

     
        cv2.imshow('frame', frame)
        cv2.waitKey(27)
        
if __name__ == '__main__':
    main()
