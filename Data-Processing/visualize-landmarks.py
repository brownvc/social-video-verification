# Visualization script
# Written by Eleanor Tursman
# Last updated 7/2020

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os
import time
import sys

inDir = sys.argv[1]

# Uncomment to use all frames instead of the first 300
# maxFrames = max( len([name for name in os.listdir(inDir + "/cam1-cropped") if os.path.isfile(os.path.join(inDir + "/cam1-cropped", name))]),
# 		 len([name for name in os.listdir(inDir + "/cam2-cropped") if os.path.isfile(os.path.join(inDir + "/cam2-cropped", name))]),
# 		 len([name for name in os.listdir(inDir + "/cam3-cropped") if os.path.isfile(os.path.join(inDir + "/cam3-cropped", name))]),
# 		 len([name for name in os.listdir(inDir + "/cam4-cropped") if os.path.isfile(os.path.join(inDir + "/cam4-cropped", name))]))

maxFrames = 300

# Real
framesCam1 = [inDir + "cam1-frames-subset/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]
framesCam2 = [inDir + "cam2-frames-subset/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)] 
framesCam3 = [inDir + "cam3-frames-subset/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]
framesCam4 = [inDir + "cam4-frames-subset/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]
framesCam5 = [inDir + "cam5-frames-subset/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]
framesCam6 = [inDir + "cam6-frames-subset/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]

landmarks1 = [inDir + "cam1-landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]
landmarks2 = [inDir + "cam2-landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]
landmarks3 = [inDir + "cam3-landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]
landmarks4 = [inDir + "cam4-landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]
landmarks5 = [inDir + "cam5-landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]
landmarks6 = [inDir + "cam6-landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]


# Fakes
# framesCam1 = [inDir + "cam1-lipgan/frames/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]
# framesCam2 = [inDir + "cam2-lipgan/frames/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)] 
# framesCam3 = [inDir + "cam3-lipgan/frames/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]
# framesCam4 = [inDir + "cam4-lipgan/frames/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]
#
# landmarks1 = [inDir + "cam1-lipgan/landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]
# landmarks2 = [inDir + "cam2-lipgan/landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]
# landmarks3 = [inDir + "cam3-lipgan/landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]
# landmarks4 = [inDir + "cam4-lipgan/landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]

for n in range(0,maxFrames):
	print(n)
	fig = plt.figure(1,figsize=(15.0, 10.0))
	plt.clf()
	
	# Camera 1
	a = fig.add_subplot(3,2,1)
	a.set_title('Camera 1')
	if not(os.path.exists(framesCam1[n])):
		print("cam1 image not found")
	else:
		plt.imshow(Image.open(framesCam1[n]))
		data = np.load(landmarks1[n],allow_pickle=True)
		if (data.size % 68 == 0):
			plt.scatter(data[:,0],data[:,1],s = 1)
		else:
			a.set_title('No face detected')
		plt.axis('off')
	
	# Camera 2
	b = fig.add_subplot(3,2,2)
	b.set_title('Camera 2')
	if not(os.path.exists(framesCam2[n])):
		print("cam2 image not found")
	else:
		plt.imshow(Image.open(framesCam2[n]))
		data = np.load(landmarks2[n],allow_pickle=True)
		if (data.size % 68 == 0):
			plt.scatter(data[:,0],data[:,1],s = 1)
		else:
			b.set_title('No face detected')
		plt.axis('off')
	
	# Camera 3
	c = fig.add_subplot(3,2,3)
	c.set_title('Camera 3') 
	if not(os.path.exists(framesCam3[n])):
		print("cam3 image not found")
	else:
		plt.imshow(Image.open(framesCam3[n]))
		data = np.load(landmarks3[n],allow_pickle=True)
		if (data.size % 68 == 0):
			plt.scatter(data[:,0],data[:,1],s = 1)
		else:
			c.set_title('No face detected')
		plt.axis('off')
	
	# Camera 4
	d = fig.add_subplot(3,2,4)
	d.set_title('Camera 4')
	if not(os.path.exists(framesCam4[n])):
		print("cam4 image not found")
	else:
		plt.imshow(Image.open(framesCam4[n]))
		data = np.load(landmarks4[n],allow_pickle=True)
		if (data.size % 68 == 0):
			plt.scatter(data[:,0],data[:,1],s = 1)
		else:
			d.set_title('No face detected')
		plt.axis('off')
		
	# Camera 5
	e = fig.add_subplot(3,2,5)
	e.set_title('Camera 5')
	if not(os.path.exists(framesCam4[n])):
		print("cam5 image not found")
	else:
		plt.imshow(Image.open(framesCam5[n]))
		data = np.load(landmarks5[n],allow_pickle=True)
		if (data.size % 68 == 0):
			plt.scatter(data[:,0],data[:,1],s = 1)
		else:
			e.set_title('No face detected')
		plt.axis('off')
	
	# Camera 6
	f = fig.add_subplot(3,2,6)
	f.set_title('Camera 6')
	if not(os.path.exists(framesCam6[n])):
		print("cam6 image not found")
	else:
		plt.imshow(Image.open(framesCam6[n]))
		data = np.load(landmarks6[n],allow_pickle=True)
		if (data.size % 68 == 0):
			plt.scatter(data[:,0],data[:,1],s = 1)
		else:
			f.set_title('No face detected')
		plt.axis('off')
		
	plt.savefig(inDir + "visualization/" + '{0:04d}'.format(n) + ".jpg",pad_inches = 0)
	plt.pause(0.001)
