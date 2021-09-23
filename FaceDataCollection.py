import cv2, os

dataset = 'dataset'
me = 'me'

# Looking for path and directories, if doesn't exists it creates
path = os.path.join(dataset, me)
if not os.path.isdir(path):
    os.makedirs(path)

# Loading algo
FaceAlgo = "E:\\FaceDetection-And-More\\Haarcascade\\haarcascade_frontalface_default.xml"
# This is a smart solution (applying limited feature out of 6000 in stages)
# used to find only the possible pixels containing face
# and then applying remaining 6000 feature in stages to the area selected
haarFaceClassifier = cv2.CascadeClassifier(FaceAlgo)

cam = cv2.VideoCapture(0)

count = 1

while count<31:

    _,img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 'detectMultiScale' is used for detecting 
    class_face = haarFaceClassifier.detectMultiScale(grayImg, 1.3, 5)

    # Cropping out face area
    for (x,y,w,h) in class_face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        FaceOnly = grayImg[y:y+h,x+w]
        # Width, Height of image
        Face_resize = cv2.resize(FaceOnly, (130,100))
        cv2.imwrite("%s/%s.jpg" %(path,count), Face_resize)
        count += 1
    cv2.imshow('img',img)
    key = cv2.waitKey(0)
    if key == 0:
        break

print("Face Data Captured")
cam.release()
cv2.destroyAllWindows



