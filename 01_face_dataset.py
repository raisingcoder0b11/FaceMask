import cv2
import os

cam = cv2.VideoCapture(0)


face_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

# For each person, enter one numeric face id
face_id = input('\n Make sure the first user entered is 0.enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        w=w+50
        h=h+50

        cv2.rectangle(img, (x,y), (x+w+50,y+h+50), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        gray = gray[y:y+h,x:x+w]
        
        cv2.imwrite("dataset/Mask/User." + str(face_id) + '.' + str(count) + ".jpg",gray )

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 70: # Take 70 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


