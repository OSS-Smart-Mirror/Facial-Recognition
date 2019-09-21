import face_recognition
import os
import cv2
import numpy as np
from flask import Flask

app = Flask(__name__)

abhay_image_1 = face_recognition.load_image_file("known_users/Abhay,1.jpg")
abhay_face_encoding_1 = face_recognition.face_encodings(abhay_image_1)[0]
abhay_image_2 = face_recognition.load_image_file("known_users/Abhay,2.jpg")
abhay_face_encoding_2 = face_recognition.face_encodings(abhay_image_2)[0]
prat_image_1 = face_recognition.load_image_file("known_users/Prat,1.jpg")
prat_face_encoding_1 = face_recognition.face_encodings(prat_image_1)[0]
prat_image_2 = face_recognition.load_image_file("known_users/Prat,2.jpg")
prat_face_encoding_2 = face_recognition.face_encodings(prat_image_2)[0]
rtvik_image_1 = face_recognition.load_image_file("known_users/Rtvik,1.jpg")
rtvik_face_encoding_1 = face_recognition.face_encodings(rtvik_image_1)[0]
rtvik_image_2 = face_recognition.load_image_file("known_users/Rtvik,2.jpg")
rtvik_face_encoding_2 = face_recognition.face_encodings(rtvik_image_2)[0]
# ishaan_image_1 = face_recognition.load_image_file("known_users/Ishaan,1.jpg")
# ishaan_face_encoding_1 = face_recognition.face_encodings(ishaan_image_1)[0]
# ishaan_image_2 = face_recognition.load_image_file("known_users/Ishaan,2.jpg")
# ishaan_face_encoding_2 = face_recognition.face_encodings(ishaan_image_2)[0]

known_face_encodings = [
   abhay_face_encoding_1,
   abhay_face_encoding_2,
   prat_face_encoding_1,
   prat_face_encoding_2,
   rtvik_face_encoding_1,
   rtvik_face_encoding_2,
   # ishaan_face_encoding_1,
   # ishaan_face_encoding_2,
]

known_face_names = [
   "Abhay",
   "Abhay",
   "Prat",
   "Prat",
   "Rtvik",
   "Rtvik",
   # "Ishaan",
   # "Ishaan"
]
face_locations = []
face_encodings = []
face_names = []

@app.route('/', methods=['GET'])
def detect_face():
# Detect Face Function, returns (name/error, success/failure)
   try:
       video_capture = cv2.VideoCapture(0)
       process_this_frame = True
       while True:
           ret, frame = video_capture.read()
           small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
           cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
           rgb_small_frame = small_frame[:, :, ::-1]
           if process_this_frame:
               face_locations = face_recognition.face_locations(rgb_small_frame)
               face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
               face_names = []
               for face_encoding in face_encodings:
                   matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                   name = "Unknown"
                   face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                   best_match_index = np.argmin(face_distances)
                   if matches[best_match_index]:
                       name = known_face_names[best_match_index]
                       return (name)
                   face_names.append(name)
           process_this_frame = not process_this_frame
   except (Warning, Exception) as e:
       return str(e), False
   finally:
       video_capture.release()

if __name__ == "__main__":
   app.run(debug=True, host='127.0.0.1', port=8080)
