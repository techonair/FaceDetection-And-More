import cv2, os

dataset = 'dataset'
me = 'me'

# Looking for path and directories, if doesn't exists it creates
path = os.path.join(dataset, me)
if not os.path.isdir(path):
    os.makedirs(path)

# Loading algo
FaceAlgo = "Haarcascade\haarcascade_frontalface_default.xml"
# This is a smart solution (applying limited feature out of 6000 in stages)
# used to find only the possible pixels containing face
# and then applying remaining 6000 features in stages to that selected area
haarFaceClassifier = cv2.CascadeClassifier(FaceAlgo)

cam = cv2.VideoCapture(0)

count = 1 

while count < 31: # captures 30 images only
    print(count)
    (_,img) = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 'detectMultiScale' for finding coordinates of features
    class_face = haarFaceClassifier.detectMultiScale(grayImg, 1.3, 4)

    # Cropping out face area
    for (x,y,w,h) in class_face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        FaceOnly = grayImg[y:y+h,x:x+w]
        # (Width, Height) of image
        Face_resize = cv2.resize(FaceOnly, (((130,100))))
        #Saving images in the directory with names
        cv2.imwrite("%s/%s.jpg" %(path,count), Face_resize)
        count += 1

    cv2.imshow('img',img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

print("Face Data Captured Successfully")
cam.release()
cv2.destroyAllWindows()




