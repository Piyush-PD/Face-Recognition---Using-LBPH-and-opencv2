
import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8') # fetching out the image
        # import pdb; pdb.set_trace()
        id = os.path.split(imagePath)[-1].split(".")[1]
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces,ids = getImagesAndLabels(path)
Unique = np.unique(ids) # ['piyush', 'sanchit'],
final_label = []
count = 0
# creating a final label which will store the label in the integer format as recognizer doesnt take the label in string so need to convert 
for i in Unique:
    for j in ids:
        if i==j:
            final_label.append(count)
    count+=1            
recognizer.train(faces, np.array(final_label))
# import pdb; pdb.set_trace()
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
