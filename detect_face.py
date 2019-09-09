import cv2
import sys
import os
from subprocess import Popen, PIPE
from time import sleep, gmtime, strftime, time

def find_face_from_camera_feed(cascPath="haarcascade_frontalface_default.xml"):
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
    start_time = time()
    while True:
        sleep(0.2)
        if (time() - start_time) > 5:
            return None, "Time limit exceeded, could not find a face in the video stream"
        elif not video_capture.isOpened():
            return None, "OpenCV could not open the webcam, please check your configurations"
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
                timestamp = strftime("%H:%M:%S", gmtime()) + ".png"
                x, y, w, h = [ v for v in f ]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255))
                sub_face = frame[y:y+h, x:x+w]
                cv2.imwrite(timestamp, sub_face)
                return timestamp, None

if __name__ == "__main__":
    image_name, error_message = find_face_from_camera_feed()
    if error_message is not None:
        print(error_message)
    else:
        try:
            p = Popen(['face_recognition', 'known_users', image_name], stdout=PIPE)
            output = p.communicate()[0]
            if p.returncode != 0:
               raise SystemError("Non-zero exit code from face recognition module")
            print(output.decode("utf-8").split(',')[1])
        except (Warning, Exception) as e:
            print(e)
        finally:
            pass
            # os.remove(image_name)
