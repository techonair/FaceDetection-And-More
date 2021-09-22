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
    FaceOnly = haarFaceClassifier.detectMultiScale(grayImg, 1.3, 5)
    



