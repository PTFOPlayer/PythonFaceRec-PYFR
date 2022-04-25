#cv2 face recognition
import cv2
import threading

#capture video
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def face_detect(faces, frame):
    
    for (x, y, w, h) in faces:
        f = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(f, 'Face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if f is not None:
            return True

def eye_detect(eyes, frame):
    for (x, y, w, h) in eyes:
        e = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(e, 'Eye', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
def body_detect(bodys, ubodys, lbodys, frame):
    for (x, y, w, h) in bodys:
        b = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(b, 'Body', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for (x, y, w, h) in ubodys:
        b = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(b, 'Body', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for (x, y, w, h) in lbodys:
        b = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(b, 'Body', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
def main():
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')

    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    
    body_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')
    upper_body_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')
    lower_body_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_lowerbody.xml')

    while True:
        _, frame = capture.read()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #detect faces
        faces = face_cascade.detectMultiScale(grayscale,  1.2,  3, minSize=(30, 30))
        
        t1 = threading.Thread(target=face_detect, args=(faces, frame))
        t1.start()
        t1.join()
        #if face_detect(faces, frame) == None:
        #    print('No face detected')
        
        #detect eyes
        eyes = eye_cascade.detectMultiScale(grayscale,  1.2,  3, minSize=(30, 30))
        
        t2 = threading.Thread(target=eye_detect, args=(eyes, frame))
        t2.start()
        t2.join()
        
        #detect bodys
        bodys = body_cascade.detectMultiScale(grayscale,  1.2,  3, minSize=(30, 30))
        ubodys = upper_body_cascade.detectMultiScale(grayscale,  1.2,  3, minSize=(30, 30))
        lbodys = lower_body_cascade.detectMultiScale(grayscale,  1.2,  3, minSize=(30, 30))
        
        t3 = threading.Thread(target=body_detect, args=(bodys, ubodys, lbodys, frame))
        t3.start()
        t3.join()
        
        cv2.imshow('frame', frame)
        cv2.imshow('grayscale', grayscale)

        key = cv2.waitKey(50)
        if key == 27:
            break
       
        
if __name__ == '__main__':
    main()