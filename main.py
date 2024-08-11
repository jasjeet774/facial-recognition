import face_recognition
import cv2
# OpenCV or cv2 is used to capture the picture or face from the webcam and then process it,
# after the processing it sent to the face_recognition so face recogniser will recognise and compare the face with the 
# faces which are  already present in the database
import numpy as np
# numpy is used to numpy arrays 
import csv
# csv files is used to handle the csv file like updating ,creating 
import os
# used to access the file like csv
from datetime import datetime

video_capture=cv2.VideoCapture(0)
jobs_image=face_recognition.load_image_file("images/122.jpg")
jobs_encoding=face_recognition.face_encodings(jobs_image)[0]

piyush_image=face_recognition.load_image_file("images/121.jpg")
piyush_encoding=face_recognition.face_encodings(piyush_image)[0]

ashneer_image=face_recognition.load_image_file("images/123.jpg")
ashneer_encoding=face_recognition.face_encodings(ashneer_image)[0]

veenita_image=face_recognition.load_image_file("images/124.jpg")
veenita_encoding=face_recognition.face_encodings(veenita_image)[0]

known_face_encoding=[
    jobs_encoding,
    piyush_encoding,
    ashneer_encoding,
    veenita_encoding

]

known_face_names=[
    "jobs",
    "Piyush bansal",
    "Ashneer groover",
    "veenita singh"

]
students=known_face_names.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f = open(current_date+'.csv',newline = '')
lnwriter=csv.writer(f)

while True:
    captured,frame= video_capture.read()
    if not captured:
        print("Not captured")
        break
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)

        face_names=[]
        for face_encoding in face_encodings:
            print(face_encoding)
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]


            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("attendance system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()    
