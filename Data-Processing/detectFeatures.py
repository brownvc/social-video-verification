# From: https://github.com/1adrianb/face-alignment

import face_alignment
from skimage import io
import numpy as np
import sys
import os

inDir = sys.argv[1]
outDir = sys.argv[2]
bounding = sys.argv[3]

# Uncropped image at a time, dlib detector
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)

boundingFile = open(bounding, "r")

numImg = len([name for name in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, name))])
for f in range(1,numImg+1):
	filename = inDir + "frames" + '{0:04d}'.format(f) + ".jpg"
	
	box = boundingFile.readline().split(',')
	boxInt = [[int(box[0]),int(box[1]), int(box[2]), int(box[3])]]
	 
	print("Processing file: {}".format(f))
	img = io.imread(filename)
	pred = fa.get_landmarks_from_image(img,detected_faces=boxInt)
	pred = np.array(pred)
	
	if (pred.size == 1):
		print("No face detected")
	else:
		curData = np.reshape(pred,(68,-1))	
	
	np.save(outDir + 'landmarks2D-' + '{0:04d}'.format(f) + '.npy',curData)


boundingFile.close()
