import cv2
import datetime
import numpy as np
import os
import collections
import cPickle as pickle
import random as rand
from sklearn import svm
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog
import time

c=0

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """ Define a function to return HOG features and visualization """
    # Call with two outputs if vis==True
    feature_image = np.array(img, dtype=np.uint8)
    # feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HLS)
    if vis == True:
        features, hog_image = hog(feature_image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(feature_image, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def getImageFeatures(image):
    features=[]
    for channel in range(image.shape[2]):
        features.extend(get_hog_features(image[:, :, channel],
                                             9, 8, 2,
                                             vis=False, feature_vec=True))
    return features


def training(positiveLocation,NegativeLocation):
    # Feature Extraction...
    positive_imgs=[]
    negative_imgs=[]
    for folder in os.listdir(positiveLocation):
        for file in os.listdir(positiveLocation+"/"+folder):
            image = os.path.join(positiveLocation+"/"+folder, file)
            img=cv2.imread(image)
            img=cv2.resize(img,(64,64))
            positive_imgs.append(img)

    for folder in os.listdir(NegativeLocation):
        for file in os.listdir(NegativeLocation + "/" + folder):
            image = os.path.join(NegativeLocation+"/"+folder, file)
            img=cv2.imread(image)
            img=cv2.resize(img,(64,64))
            negative_imgs.append(img)

    print "Extracting HOG features from positive and negative images"
    positive_features=[]
    negative_features=[]
    for img in positive_imgs:
        f=getImageFeatures(img)
        positive_features.append(f)
    for img in negative_imgs:
        f=getImageFeatures(img)
        negative_features.append(f)
    print "Features Extracted."
    # Scaling Features...



    unscaled_x = np.vstack(( positive_features, negative_features)).astype(np.float64).squeeze()
    scaler = StandardScaler().fit(unscaled_x)
    x = scaler.transform(unscaled_x)
    y = np.hstack((np.ones(len(positive_imgs)), np.zeros(len(negative_imgs))))
    print "Initiating training."
    # Training Features...
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = rand.randint(0, 100))
    svc = LinearSVC()
    svc.fit(x_train, y_train)
    accuracy = svc.score(x_test, y_test)
    pickle.dump((svc,scaler), open("vehicleSVC.p", "wb"))
    print "Accuracy:"+str(accuracy)

def is_TrafficLight(img):
    classifier = pickle.load(open('./trafficLights.p', 'rb'))
    features = getHOGFeatures(img)

    if classifier.predict(np.array(features).T) == 1:
        return True
    else:
        return False

def cropImage(image,x,y,w,h):
    crop_img = image[y:y+h, x:x+w]  # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    return crop_img



def PreProcessTrafficBox(image):
    lower_yellow= np.array([16, 100, 100])
    upper_yellow= np.array([45, 255, 255])

    lower_red1= np.array([0, 0 , 0])
    upper_red1= np.array([10, 250, 250])

    lower_red2= np.array([300, 0 , 0])
    upper_red2= np.array([359, 255, 255])

    lower_green= np.array([60,0,0])
    upper_green= np.array([100,255,255])

    hsv1= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv1, lower_yellow, upper_yellow)
    yellowRemoved = cv2.bitwise_not(hsv1,hsv1,mask = mask)


    mask2 = cv2.inRange(yellowRemoved, lower_red1, upper_red1)
    mask3 = cv2.inRange(yellowRemoved, lower_red2, upper_red2)
    mask4 = cv2.inRange(yellowRemoved, lower_green, upper_green)
    redRetained = cv2.bitwise_and(yellowRemoved,yellowRemoved,mask = (mask2+mask3+mask4))


    # cv2.imshow("orginal",image)
    # cv2.imshow("red",totalMask)

    res=cv2.cvtColor(redRetained,cv2.COLOR_HSV2BGR)
    # cv2.imshow("masked",res)

    grayScaleImage = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    thresh, im_bw = cv2.threshold(grayScaleImage, 90, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Cropped Image",image)
    # cv2.waitKey()
    # cv2.imshow("Thresholded Image",im_bw)
    # cv2.waitKey()
    return im_bw

def DecideTrafficLight(image):
    # cv2.imshow("Traffic Light", image)
    height, width = image.shape
    print width, height
    WhitePixelsCountByRegion = []
    heightRegion=height/3
    for i in range(3):
        lowerLimit=i*heightRegion
        heightLimit=(i+1)*heightRegion
        heightLimit=height-1 if heightLimit>=height else heightLimit
        count_white = cv2.countNonZero(image[lowerLimit:heightLimit])
        WhitePixelsCountByRegion.append(count_white)
    print "FirstRegion:",WhitePixelsCountByRegion[0]
    print "SecondRegion:", WhitePixelsCountByRegion[1]
    print "ThirdRegion:", WhitePixelsCountByRegion[2]
    maxPixelcount=max(WhitePixelsCountByRegion)
    print WhitePixelsCountByRegion.index(maxPixelcount)+1
    print "------------------------------------------------"
    return WhitePixelsCountByRegion.index(maxPixelcount)+1



def checkNearBySquares(x1,y1,x2,y2,distance):
    if (x1-x2)**2 + (y1-y2)**2 < distance**2:
        return True
    else:
        return False


def FindTrafficLight(image):
    found=False
    hsv1=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # define range of red color in HSV
    # lower_red = np.array([10, 100, 50])
    # upper_red = np.array([15, 255, 255])

    lower_yellow= np.array([15, 60, 60])
    upper_yellow= np.array([30, 255, 255])


    mask = cv2.inRange(hsv1, lower_yellow, upper_yellow)

    res = cv2.bitwise_and(image,image,mask = mask)
    grayScaleImage= cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Converted to GrayScale",grayScaleImage)
    thresh, im_bw = cv2.threshold(grayScaleImage,155,255,cv2.THRESH_BINARY)
    # cv2.imshow("Applied Threshold",im_bw)



    # cv2.imshow("bitwise mask",res)
    # mask=cropImage(mask)

    edges = cv2.Canny(im_bw,225,225,apertureSize = 3)
    # cv2.imshow("edges",edges)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



    #cv2.imshow("thresholding",im_bw)
    rectangularContours = []
    greatestArea = 0
    rectangleList = []

    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(contour)
        ratio = float(h)/float(w)
        area = (w*h) #and (area>=1000)  and (ratio>=1.4)

        if y<350 and (area>=1000)  and  (1500 > x> 700) and (ratio>=1.4):
            rectangularContours.append([x, y, w, h])
            ep = datetime.datetime(1970, 1, 1, 0, 0, 0)
            X = (datetime.datetime.utcnow() - ep).total_seconds()
            box = cropImage(image, x+2, y+2, w-2, h-2)
            # cv2.imwrite("./images/output/box"+str(X)+".jpg",box)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    finalRectangles=collections.defaultdict(list)
    counter=0

    for currentX, currentY, currentWidth, currentHeight in rectangularContours:
        rectangle=[currentX,currentY,currentWidth,currentHeight]
        currentRatio = float(currentHeight)/float(currentWidth)
        # for x, y, w, h in rectangularContours:
        #     compareRatio = float(h)/float(w)
        #     if checkNearBySquares(currentX, currentY, x, y, 15) and (currentWidth*currentHeight) > (w*h) and (currentRatio>compareRatio):
        #         rectangle=[x, y, w, h]
        currentWidth = currentHeight/2;
        croppedImage = cropImage(image, currentX, currentY, currentWidth, currentHeight)

        if (is_TrafficLight(croppedImage)):
            cv2.rectangle(image, (currentX, currentY), (currentX + currentWidth, currentY + currentHeight), (0, 255, 0), 2)
            found=True
            break
        # else:
        #     cv2.rectangle(image, (currentX, currentY), (currentX + currentWidth, currentY + currentHeight), (0, 0, 0), 2)

    return image,found


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    img_Copy=np.copy(image)
    detectedRectangleList=[]
    global c
    for y in xrange(500, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            if(x+windowSize[1]>image.shape[1] or y + windowSize[1]>image.shape[0] ):
                break
            cropped_imge=cropImage (image,x, y,windowSize[1],windowSize[0])

            # cv2.imwrite("./images/data/image"+str(c)+".png",cropped_imge)
            c+=1
            # cv2.rectangle(img_Copy, (x, y), (x + windowSize[0], y + windowSize[1]),(0, 255, 0), 2)
            # cv2.imshow("Window search",img_Copy)
            # cv2.waitKey(10)
            if DetectCar(cropped_imge):
                cv2.rectangle(img_Copy,(x,y),(x + windowSize[0],y + windowSize[1]),(255, 0, 0), 3)
                detectedRectangleList.append([x,y,(x + windowSize[0]),(y + windowSize[1])])

    return img_Copy

def DetectCar(image):
    classifier, X_scaler = pickle.load(open('./vehicleSVC.p', 'rb'))
    test_img=cv2.resize(image,(64,64))
    features = getImageFeatures(test_img)
    test_features = X_scaler.transform(np.array(features).reshape(1, -1))

    if classifier.predict(test_features) == 1:
        return True
    else:
        return False



positive="./images/OwnCollection/OwnCollection/vehicles"
negative="./images/OwnCollection/OwnCollection/non-vehicles"
training(positive,negative)
#
# img=cv2.imread("./images/Car.jpg")
# output=sliding_window(img,50, [128,128])
# cv2.imwrite("./Output.jpg",output)
# cv2.waitKey()
# inputPath="./images/1"
# outputPath="./images/output"
# for file in os.listdir(inputPath):
#     image=os.path.join(inputPath, file)
#     img=cv2.imread(image)
#     output = sliding_window(img, 50, [128, 128])
#     filename=str(file.replace(".jpg",""))+".jpg"
#     cv2.imwrite(os.path.join(outputPath,filename),output)




