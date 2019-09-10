import cv2
import sys
import os
from subprocess import Popen, PIPE
from time import sleep, gmtime, strftime, time

def find_face_from_camera_feed(cascPath="haarcascade_frontalface_default.xml"):
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
    sleep(0.5)
    start_time = time()
    while True:
        if (time() - start_time) > 30:
            raise TimeoutError("Operation timed out")
        elif not video_capture.isOpened():
            raise cv2.error("Camera could not be opened")
        else:
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            for f in faces:
                user = None
                try:
                    image_name = strftime("%H:%M:%S", gmtime()) + ".png"
                    x, y, w, h = [ v for v in f ]
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255))
                    sub_face = frame[y:y+h, x:x+w]
                    cv2.imwrite(image_name, sub_face)
                    p = Popen(['face_recognition', 'known_users', image_name], stdout=PIPE)
                    output = p.communicate()[0]
                    if (p.returncode != 0):
                       raise SystemError("Non-zero exit code from face recognition module")
                    elif "no_persons_found" in output.decode("utf-8"):
                        continue
                    else:
                        return output.decode("utf-8").split(',')[1], None
                except (Warning, Exception) as e:
                    return None, e
                finally:
                    os.remove(image_name)

# def run_face_detect():
#     image_name = find_face_from_camera_feed()
#     p = Popen(['face_recognition', 'known_users', image_name], stdout=PIPE)
#     output = p.communicate()[0]
#     if p.returncode != 0:
#        raise SystemError("Non-zero exit code from face recognition module")
#     return output.decode("utf-8").split(',')[1]
#     os.remove(image_name)

def detect_face():
    print(find_face_from_camera_feed())

detect_face()
