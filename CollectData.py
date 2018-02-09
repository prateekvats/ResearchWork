import cv2
import os
import numpy as np


c=0
windowSizes=[150]
imagesPath="./images/DataProcessing/collectedData"

def cropImage(image,x,y,w,h):
    crop_img = image[y:y+h, x:x+w]  # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    return crop_img

def sliding_window_crop_save(image, stepSize, windowSize):
    # slide a window across the image
    global c
    for y in xrange(image.shape[0]/2, image.shape[0]-100, stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            if(x+windowSize[1]>image.shape[1] or y + windowSize[1]>image.shape[0] ):
                break
            cropped_imge=cropImage(image,x, y,windowSize[1],windowSize[0])

            cv2.imwrite("./images/DataProcessing/objectsFromImages/"+str(c)+".png",cropped_imge)
            c+=1
            # cv2.rectangle(img_Copy, (x, y), (x + windowSize[0], y + windowSize[1]),(0, 255, 0), 2)
            # cv2.imshow("Window search",img_Copy)
            # cv2.waitKey(10)



def collectData(rawPath):
    totalFilesCount=len([name for name in os.listdir(rawPath)])
    doneFiles=0
    for file in os.listdir(rawPath):
        image = os.path.join(rawPath, file)
        img=cv2.imread(image)
        for size in windowSizes:
            sliding_window_crop_save(img,20,[size,size])
        print float(doneFiles)/float( totalFilesCount)*100," % complete"
        doneFiles+=1




collectData(imagesPath)



