import sys
import os
import numpy as np
import itertools
from matplotlib import pyplot as plt
import scipy.io as sio

def framewiseL2(a,b):
	frameL2 = np.linalg.norm(a - b, ord=2)
	return frameL2

def l2Comparison(inPath,numCams,outPath,numF,fakeCam,shift):
	# FLAME figure
	figF1 = plt.figure()
	figF2 = plt.figure()
	figF3 = plt.figure()
	axF1 = figF1.add_subplot(111)
	axF2 = figF2.add_subplot(111)
	axF3 = figF3.add_subplot(111)
	
	# 3DMM figure
	figB1 = plt.figure()
	axB1 = figB1.add_subplot(111)
	figB2 = plt.figure()
	axB2 = figB2.add_subplot(111)
	figB3 = plt.figure()
	axB3 = figB3.add_subplot(111)

	# One versus all L2 comparison
	# Look at all combinations of cameras when computing framewise L2
	for j in range(1,numCams+1):
	
		flameShape = np.zeros((numF,numCams))
		flameExp = np.zeros((numF,numCams))
		flameAll = np.zeros((numF,numCams))
	
		baselShape = np.zeros((numF,numCams))
		baselExp = np.zeros((numF,numCams))
		baselAll = np.zeros((numF,numCams))
		
		for k in range(1,numCams+1):
			if(j == k):
				continue
			# Iterate over frames
			for i in range(1,numF + 1):
		
				# For camera 7, use faked version of camera fakeCam
				if(j == numCams):
					flame1 = np.load(inPath + ('cam%d-lipgan/FLAME/' % fakeCam) + '{0:04d}'.format(i) + '_params.npy', allow_pickle=True).tolist()
					basel1 = np.load(inPath + ('cam%d-lipgan/3DMM/' % fakeCam) + '{0:04d}'.format(i) + '_params.npy', allow_pickle=True).tolist() 
				else:
					flame1 = np.load(inPath + ('cam%d-FLAME/' % j) + '{0:04d}'.format(i+shift) + '_params.npy', allow_pickle=True).tolist()
					basel1 = np.load(inPath + ('cam%d-3DMM/' % j) + '{0:04d}'.format(i) + '_params.npy', allow_pickle=True).tolist()
				if(k == numCams):
					flame2 = np.load(inPath + ('cam%d-lipgan/FLAME/' % fakeCam) + '{0:04d}'.format(i) + '_params.npy', allow_pickle=True).tolist()
					basel2 = np.load(inPath + ('cam%d-lipgan/3DMM/' % fakeCam) + '{0:04d}'.format(i) + '_params.npy', allow_pickle=True).tolist()
				else:
					flame2 = np.load(inPath + ('cam%d-FLAME/' % k) + '{0:04d}'.format(i+shift) + '_params.npy', allow_pickle=True).tolist()
					basel2 = np.load(inPath + ('cam%d-3DMM/' % k) + '{0:04d}'.format(i) + '_params.npy', allow_pickle=True).tolist()
	
				# Flame shape     
				fShape1 = flame1["shape"][:]
				fShape2 = flame2["shape"][:]
				flameShape[i-1,k-1] = framewiseL2(fShape1, fShape2)
		
				# Flame expr (+ pose)
				# Pose 1-3: neck, 4-6: jaw, 7-9: eyes (left), 10-12: eyes (right)
				fExp1 = np.hstack((flame1["exp"][:], flame1["pose"][3:6]))
				fExp2 = np.hstack((flame2["exp"][:], flame2["pose"][3:6]))
				flameExp[i-1,k-1] = framewiseL2(fExp1, fExp2)
	
				# Flame shape + expr
				flameAll[i-1,k-1] = framewiseL2(np.hstack((fShape1,fExp1)), np.hstack((fShape2,fExp2)))
	
				# # 3DMM shape
				bShape1 = basel1["shape"][:]
				bShape2 = basel2["shape"][:]
				baselShape[i-1,k-1] = framewiseL2(bShape1, bShape2)
	
				# # 3DMM expr
				bExp1 = basel1["exp"][:]
				bExp2 = basel2["exp"][:]
				baselExp[i-1,k-1] = framewiseL2(bExp1, bExp2)
				
				# # 3DMM shape + expr
				baselAll[i-1,k-1] = framewiseL2(np.vstack((bShape1,bExp1)), np.vstack((bShape2,bExp2)))
		
		# Remove zero col from where j == k
		flameShape = np.delete(flameShape,j-1,axis=1)
		flameExp = np.delete(flameExp,j-1,axis=1)
		flameAll = np.delete(flameAll,j-1,axis=1)
		
		baselShape = np.delete(baselShape,j-1,axis=1)
		baselExp = np.delete(baselExp,j-1,axis=1)
		baselAll = np.delete(baselAll,j-1,axis=1)
		
		# Take average across cameras per frame
		meanFShape = np.mean(flameShape,axis=1)
		stdFShape = np.std(flameShape,axis=1)
		
		meanFExp = np.mean(flameExp,axis=1)
		stdFExp = np.std(flameExp,axis=1)

		meanFAll = np.mean(flameAll,axis=1)
		stdFAll = np.std(flameAll,axis=1)

		meanBShape = np.mean(baselShape,axis=1)
		stdBShape = np.std(baselShape,axis=1)

		meanBExp = np.mean(baselExp,axis=1)
		stdBExp = np.std(baselExp,axis=1)

		meanBAll = np.mean(baselAll,axis=1)
		stdBAll = np.std(baselAll,axis=1)

		# Plotting
		if (j == numCams):
			axF1.plot(range(1,numF+1), meanFShape, label="Fake Camera",color='r')
			axF1.fill_between(range(1,numF+1), meanFShape - stdFShape, meanFShape + stdFShape, color='r', alpha=0.4)
			print("camera: fake, std:", np.mean(stdFShape))

			axF2.plot(range(1,numF+1), meanFExp, label="Fake Camera",color='r')
			axF2.fill_between(range(1,numF+1), meanFExp - stdFExp, meanFExp + stdFExp, color='r', alpha=0.4)
			print("camera: fake, std:", np.mean(stdFExp))

			axF3.plot(range(1,numF+1), meanFAll, label="Fake Camera",color='r')
			axF3.fill_between(range(1,numF+1), meanFAll - stdFAll, meanFAll + stdFAll, color='r', alpha=0.4)
			print("camera: fake, std:", np.mean(stdFAll))

			axB1.plot(range(1,numF+1), meanBShape, label="Fake Camera",color='r')
			axB1.fill_between(range(1,numF+1), meanBShape - stdBShape, meanBShape + stdBShape, color='r', alpha=0.4)

			axB2.plot(range(1,numF+1), meanBExp, label="Fake Camera",color='r')
			axB2.fill_between(range(1,numF+1), meanBExp - stdBExp, meanBExp + stdBExp, color='r', alpha=0.4)

			axB3.plot(range(1,numF+1), meanBAll, label="Fake Camera",color='r')
			axB3.fill_between(range(1,numF+1), meanBAll - stdBAll, meanBAll + stdBAll, color='r', alpha=0.4)
		else:
			axF1.plot(range(1,numF+1), meanFShape, label="Camera %d vs all" % j)
			axF1.fill_between(range(1,numF+1), meanFShape - stdFShape, meanFShape + stdFShape, alpha=0.2)
			print("camera: %d, std:" % (j), np.mean(stdFShape))

			axF2.plot(range(1,numF+1), meanFExp, label="Camera %d vs all" % j)
			axF2.fill_between(range(1,numF+1), meanFExp - stdFExp, meanFExp + stdFExp, alpha=0.2)
			print("camera: %d, std:" % (j), np.mean(stdFExp))

			axF3.plot(range(1,numF+1), meanFAll, label="Camera %d vs all" % j)
			axF3.fill_between(range(1,numF+1), meanFAll - stdFAll, meanFAll + stdFAll, alpha=0.2)
			print("camera: %d, std:" % (j), np.mean(stdFAll))

			axB1.plot(range(1,numF+1), meanBShape, label="Camera %d vs all" % j)
			axB1.fill_between(range(1,numF+1), meanBShape - stdBShape, meanBShape + stdBShape, alpha=0.2)

			axB2.plot(range(1,numF+1), meanBExp, label="Camera %d vs all" % j)
			axB2.fill_between(range(1,numF+1), meanBExp - stdBExp, meanBExp + stdBExp, alpha=0.2)

			axB3.plot(range(1,numF+1), meanBAll, label="Camera %d vs all" % j)
			axB3.fill_between(range(1,numF+1), meanBAll - stdBAll, meanBAll + stdBAll, alpha=0.2)
	
	axF1.set_xlabel('Frame')
	axF1.set_ylabel('Mean Framewise L2 Difference Across Cameras')
	axF1.set_title("Flame Shape one vs all")
	axF1.legend()
	
	axF2.set_xlabel('Frame')
	axF2.set_ylabel('Mean Framewise L2 Difference Across Cameras')
	axF2.set_title("Flame Exp one vs all")
	axF2.legend()
	
	axF3.set_xlabel('Frame')
	axF3.set_ylabel('Mean Framewise L2 Difference Across Cameras')
	axF3.set_title("Flame Shape+Exp one vs all")
	axF3.legend()

	axB1.set_xlabel('Frame')
	axB1.set_ylabel('Mean Framewise L2 Difference Across Cameras')
	axB1.set_title("3DMM Shape one vs all")
	axB1.legend()

	axB2.set_xlabel('Frame')
	axB2.set_ylabel('Mean Framewise L2 Difference Across Cameras')
	axB2.set_title("3DMM Exp one vs all")
	axB2.legend()

	axB3.set_xlabel('Frame')
	axB3.set_ylabel('Mean Framewise L2 Difference Across Cameras')
	axB3.set_title("3DMM Shape+Exp one vs all")
	axB3.legend()
	
	figF1.savefig(outPath + 'flame-shape.png')
	figF2.savefig(outPath + 'flame-exp.png')
	figF3.savefig(outPath + 'flame-shape-exp.png')
	figB1.savefig(outPath + '3dmm-shape.png')
	figB2.savefig(outPath + '3dmm-exp.png')
	figB3.savefig(outPath + '3dmm-shape-exp.png')

if __name__ == '__main__':
	
	person = sys.argv[2] 				#"ID6"
	fakeCam = int(sys.argv[1]) 			#Fake camera number, eg: 1, 2, 3, 4, 5, 6
	inPath = sys.argv[3] + person + "/"
	numCams = 7 						# six real cameras + one fake
	outPath = inPath + "experiment-results/"
	print("Saving output figures to: {}".format(outPath))

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

	# running on the first 25 frames
	l2Comparison(inPath,numCams,outPath,25,fakeCam,shift)