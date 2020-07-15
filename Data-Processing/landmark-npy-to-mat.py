import sys
import os
import numpy as np
import scipy.io as sio

def landmark2mat(inPath,numCams,outPath,numF,fakeCam,shift):
# Prep landmark data for analysis in Matlab
# Lip points are: 49-68

	numLip = 20
	dataMat = np.zeros((numF,numLip*2,numCams))

	# Calculate data matrix of landmarks for each camera
	for i in range(1,numCams+1):
		
		means = np.zeros((numLip*2,numF))

		# Get feature location per frame in image space
		for f in range(1,numF+1):
			# Load data
			# format: x1,y1, x2,y2, ... xn,yn, for lip points 1,...,n
			if (i == numCams):
				data = np.load(inPath + ('cam%d-lipgan/landmarks/landmarks2D-' % fakeCam) + '{0:04d}'.format(f) + '.npy', allow_pickle=True)
			else:
				data = np.load(inPath + ('cam%d-landmarks/landmarks2D-' % i) + '{0:04d}'.format(f+shift) + '.npy', allow_pickle=True)
					
			means[:,f-1] = np.squeeze(np.reshape(data[-numLip:,:],(-1,1)))

		# Need it to be in format where observations are rows, vars are cols		
		camData = means.T

		# Normalize data for camera i, which 0-indexed is i-1
		camMean = np.mean(camData,axis=0)
		camStd = np.std(camData,axis=0)
		
		dataMat[:,:,i-1] = (camData - camMean) / camStd

	# Save data matrix as mat
	sio.savemat(outPath,{'cam1':dataMat[:,:,0],'cam2':dataMat[:,:,1],'cam3':dataMat[:,:,2],'cam4':dataMat[:,:,3],
						 'cam5':dataMat[:,:,4],'cam6': dataMat[:,:,5],'fake':dataMat[:,:,6]})

if __name__ == '__main__':

	fakeCam = int(sys.argv[1])          #eg: 4
	person = sys.argv[2]                #eg: "ID6"
	inPath = sys.argv[3] + person + "/"
	numCams = 7                         # six real cameras + one fake
	outPath = inPath + ("experiment-results/mouth-data-fake%d-" % fakeCam) + person + ".mat"
	print("Saving output to: {}".format(outPath))

	# LipGAN's output is shifted from the input by five frames
	shift = 5	    

	# Take the shortest number of frames available
	numF = min(len([name for name in os.listdir(inPath + ('cam%d-lipgan/landmarks/' % fakeCam)) if os.path.isfile(os.path.join(inPath +  ('cam%d-lipgan/landmarks/' % fakeCam), name))]),
			   len([name for name in os.listdir(inPath + 'cam1-landmarks/') if os.path.isfile(os.path.join(inPath + 'cam1-landmarks/', name))]) - shift,
			   len([name for name in os.listdir(inPath + 'cam2-landmarks/') if os.path.isfile(os.path.join(inPath + 'cam2-landmarks/', name))]) - shift,
			   len([name for name in os.listdir(inPath + 'cam3-landmarks/') if os.path.isfile(os.path.join(inPath + 'cam3-landmarks/', name))]) - shift,
			   len([name for name in os.listdir(inPath + 'cam4-landmarks/') if os.path.isfile(os.path.join(inPath + 'cam4-landmarks/', name))]) - shift,
			   len([name for name in os.listdir(inPath + 'cam5-landmarks/') if os.path.isfile(os.path.join(inPath + 'cam5-landmarks/', name))]) - shift,
			   len([name for name in os.listdir(inPath + 'cam6-landmarks/') if os.path.isfile(os.path.join(inPath + 'cam6-landmarks/', name))]) - shift)

	landmark2mat(inPath,numCams,outPath,numF,fakeCam, shift)
