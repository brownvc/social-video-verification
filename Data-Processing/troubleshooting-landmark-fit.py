# Visualization script
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os
import time
import sys

inDir = sys.argv[1]

maxFrames = len([name for name in os.listdir(inDir + "cam3-lipgan/frames/") if os.path.isfile(os.path.join(inDir + "cam3-lipgan/frames/", name))])

# Real
shift = 5
framesCam3R = [inDir + "cam3-frames-subset/frames" + '{0:04d}'.format(i+shift) + ".jpg" for i in range(1,maxFrames+1)]
landmarks3R = [inDir + "cam3-landmarks/landmarks2D-" + '{0:04d}'.format(i+shift) + ".npy" for i in range(1,maxFrames+1)]

# Fakes
framesCam3F = [inDir + "cam3-lipgan/frames/frames" + '{0:04d}'.format(i) + ".jpg" for i in range(1,maxFrames+1)]
landmarks3F = [inDir + "cam3-lipgan/landmarks/landmarks2D-" + '{0:04d}'.format(i) + ".npy" for i in range(1,maxFrames+1)]


for n in range(0,maxFrames):
	print(n)
	fig = plt.figure(1,figsize=(15.0, 10.0))
	plt.clf()
	
	# Camera 3
	c = fig.add_subplot(1,2,1)
	c.set_title('Real Camera 3') 
	if not(os.path.exists(framesCam3R[n])):
		print("cam3 image not found")
	else:
		plt.imshow(Image.open(framesCam3R[n]))
		data = np.load(landmarks3R[n],allow_pickle=True)
		if (data.size % 68 == 0):
			plt.scatter(data[:,0],data[:,1],s = 1)
		else:
			c.set_title('No face detected')
		plt.axis('off')
	
	d = fig.add_subplot(1,2,2)
	d.set_title('Fake Camera 3')
	if not(os.path.exists(framesCam3F[n])):
		print("cam4 image not found")
	else:
		plt.imshow(Image.open(framesCam3F[n]))
		data = np.load(landmarks3F[n],allow_pickle=True)
		if (data.size % 68 == 0):
			plt.scatter(data[:,0],data[:,1],s = 1)
		else:
			d.set_title('No face detected')
		plt.axis('off')
				
	plt.savefig(inDir + "visualization2/" + '{0:04d}'.format(n) + ".jpg",pad_inches = 0)
	plt.pause(0.001)
