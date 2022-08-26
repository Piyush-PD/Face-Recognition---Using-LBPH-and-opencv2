
import cv2
import os

cam = cv2.VideoCapture(0) # <-- here 0 means default camera (main camera), if other camera or extra camera is present then it will be 1
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id')

print("\n [INFO] Initializing face capture....")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read() # ret is TRUE as camera is active and img is the array
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting the image to grey scale for fast processing and optimization
    faces = face_detector.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5) # Here we are using harrascade to detect the face area from the image i.e gray
    # refer to documentation for the explaination -> https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  # Getting out the diagonal point for it....  
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff 
    if k == 27: 
        break
    elif count >= 30: # if dataset reaches to 30 images then it will break the loop 
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()


