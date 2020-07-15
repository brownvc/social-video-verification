# Written by Eleanor Tursman
# Last updated 7/2020
# Commented lines can be used to debug

import numpy as np 
import cv2
from skimage import io
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import sys
import os

camID = int(sys.argv[1])
inDir = sys.argv[2] + ("cam%d" % camID) + "-frames/"
features = sys.argv[2] + ("cam%d" % camID) + "-landmarks/"
boundingFile = open(sys.argv[2] + "bounding-boxes/" + ("cam%d" % camID) + "-bounding-boxes.txt",'r')
boundingFile = open(sys.argv[2] + "bounding-boxes/" + ("cam%d" % camID) + "-lipgan-bounding-boxes.txt","w+")
numFrames =  len([name for name in os.listdir(features) if os.path.isfile(os.path.join(features, name))])

for f in range(1,numFrames+1):
    

    #box = boundingFile.readline().split(',')
    #boxInt = [[int(box[0]),int(box[1]), int(box[2]), int(box[3])]]
    print("Processing file: {}".format(f))


    #filename = inDir + "frames" + '{0:04d}'.format(f) + ".jpg"
    #img = cv2.imread(filename)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    data = np.load(features + "landmarks2D-" + '{0:04d}'.format(f) + ".npy",allow_pickle=True)

    #fig,ax = plt.subplots(1)
    #plt.imshow(img)
    #plt.scatter(data[:,0],data[:,1],s = 1,c='b')

    # Build bounding box around landmarks
    x1 = min(data[:,0])
    y1 = min(data[:,1])
    x2 = max(data[:,0])
    y2 = max(data[:,1])

    boundingFile.write('%d, %d, %d, %d\n' % (x1, y1, x2, y2))

    #rect = patches.Rectangle((boxInt[0][0],boxInt[0][1]),boxInt[0][2] - boxInt[0][0],boxInt[0][3] - boxInt[0][1],edgecolor='r',facecolor='none')  
    #rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,edgecolor='r',facecolor='none')
    #ax.add_patch(rect)

    #plt.savefig(outDir +'{0:04d}'.format(f) + '.jpg')
   
    #im2 = cv2.imread(outDir + 'tmp.jpg')

    #cv2.imwrite(outDir +'{0:04d}'.format(f) + '.jpg', cv2.rectangle(im2,(boxInt[0][0],boxInt[0][1]),(boxInt[0][2],boxInt[0][3]),(255,0,0)))

boundingFile.close()