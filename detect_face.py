import cv2
import sys
from time import sleep, gmtime, strftime, time

def find_face_from_camera_feed(cascPath="haarcascade_frontalface_default.xml"):
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
    start_time = time()
    while True:
        sleep(0.2)
        if (time() - start_time) > 5:
            return None, RuntimeError('Time exceeded, could not authenticate user', )
        elif not video_capture.isOpened():
            return None, RuntimeError('OpenCV could not open camera, please check your configurations')
        else:
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            if len(faces) is not 0:
                timestamp = strftime("%H:%M:%S", gmtime()) + ".png"
                cv2.imwrite(timestamp, frame)
                return timestamp, None

if __name__ == "__main__":
    image_name, error = find_face_from_camera_feed()
    print(image_name, error)
