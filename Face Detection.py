import cv2, os

dataset = 'dataset'
me = 'me'

#Looking for path and directories, if doesn't exists it creates
path = os.path.join(dataset, me)
if not os.path.isdir(path):
    os.makedirs(path)

FaceAlgo = "E:\\FaceDetection-And-More\\Haarcascade\\haarcascade_frontalface_default.xml"
haarFaceClassifier = cv2.CascadeClassifier(FaceAlgo)

cam = cv2.VideoCapture(0)

count = 0




