#cv2 face recognition
import cv2
import concurrent.futures
import math
import time

#capture video
capture = cv2.VideoCapture(0)
width = 640
height = 480
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

prev_frame_time = 0
new_frame_time = 0

def face_detect(faces, frame):
    
    for (x, y, w, h) in faces:
        f = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(f, 'Face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #draw line from center to face
        #cv2.line(frame,(x ,y),(320, 240),(0,255,0),1)
        # calculate vector from center to face
        vector_x = x - 320
        vector_y = y - 240
        # calculate length of vector
        length = math.sqrt(vector_x**2 + vector_y**2)
        # calculate angle
        angle = math.atan(vector_y/vector_x)
        # draw line from center to face
        cv2.line(frame,(320,240),(320+int(vector_x), 240+int(vector_y)),(0,255,0),1)
        cv2.line(frame,(320,240),(320+int(vector_x), 240+int(vector_y)),(0,255,0),1)
        cv2.circle(frame, (320, 240), int(length), (0, 255, 0), 2)
        cv2.circle(frame, (320, 240), int(length), (0, 255, 0), 2)
        
        #print solutions
        print("vector_x:", vector_x, "vector_y:", vector_y)
        print("length:", length)
        print("angle:", angle)


        
def main():
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        _, frame = capture.read()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #detect faces
        faces = face_cascade.detectMultiScale(grayscale,  1.2,  3, minSize=(30, 30))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = executor.map(face_detect( faces, frame))
        

        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        
        fps = round(float(fps), 2)
        fps = str(fps)
        cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('frame', frame)
        cv2.imshow('grayscale', grayscale)

        
        key = cv2.waitKey(50)
        if key == 27:
            break
       
        
if __name__ == '__main__':
    main()