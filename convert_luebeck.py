######################################################
##				 Luebeck dataset conversion				        ##
######################################################

# Transforms the raw Luebeck data (videos and gaze information) into a dataset of 
# input and output images (png files) for use training a neural network model.

# Video format
# data/movies-m4v/*.m4v
# 33.33 ms framerate (30 fps)

# Gaze format
# data/gaze/natural_movies_gaze/###_*.coord
# 4 ms sample rate (250 hz)

# Output format
# data/luebeck/###_movie_frame_in.png
# data/luebeck/###_movie_frame_out_##.png // where ## is 50/100/200 ms into the future

# Internal function:
#. The code loads an entire video and then loads one by one the viewing data for each 
#. of the subjects. The code then averages the viewing data to the framerate of the 
#. videos assigning viewing to the nearest neighbors. Finally 

import cv2
import numpy as np
import skimage as sk
from skimage.io import imsave

videos = ['beach']
observers = ['AAF']

# all crop sizes are downsampled to the middle
cropSizes = [128,256,512]

# we will look 200 ms in the future to generate the saccade map
saccadeTime = 200
# minimum pixel jump to allow the algorithm to run
minJump = 25

# HELPER FUNCTIONS

def averageGaze(gaze,time,tmin,tmax):
	times = gaze[:,0]/1000
	return np.mean(gaze[(times>(time+tmin))*(times<(time+tmax)),1:3],axis=0)

def gazeMap(gaze,x,y,ctime,ftime,tmin,tmax,blur):
	# Create a matrix of size y*x and set the positions that have (future) gazes
	# to one. Blur the map if requested. The map is relative to the current gaze
	# position.
	cgaze = averageGaze(gaze,ctime,tmin,tmax)
	# Create a np array of the correct size
	gaze_map = np.zeros((y,x),dtype='uint8')
	# For each value of gaze 
	times = gaze[:,0]/1000
	idxs = (times>ftime+tmin)*(times<ftime+tmax)
	future_gazes = gaze[idxs,1:3]
	future_gazes = np.round(future_gazes-cgaze+x/2).astype('int32')
	future_gazes[future_gazes<0] = 0
	future_gazes[future_gazes>=x] = x-1
	for row in future_gazes:
		gaze_map[row[0],row[1]]=gaze_map[row[0],row[1]]+1
	gaze_map = (gaze_map*255/np.sum(gaze_map)).astype('uint8')
	return gaze_map

def getCrop(frame,cgaze,sz):
	# pad frame
	pad = int(sz/2)
	pframe = sk.util.pad(frame,((pad,pad),(pad,pad),(0,0)),'constant')
	# crop
	cgaze = np.round(cgaze+pad).astype('int')
	crop = pframe[(cgaze[0]-pad):(cgaze[0]+pad),(cgaze[1]-pad):(cgaze[1]+pad),:]
	# scikit image probably has a simple way of doing this
	return crop

# MAIN CODE

for video in videos:

	datas = {}
	# load the observers gaze files
	for obs in observers:
		print('Loading observer gaze data: '+ obs)
		datas[obs] = np.loadtxt('data/gaze/natural_movies_gaze/'+obs+'_'+video+'.coord', skiprows=2)

	# For each video load one frame at a time
	fname = 'data/movies-m4v/'+video+'.m4v'
	print('Loading video file: ' + fname)
	cap = cv2.VideoCapture(fname)

	# Use the frames per second to generate a window to average gaze
	# data over
	fps = cap.get(cv2.CAP_PROP_FPS)
	gmin = -fps/2
	gmax = -gmin

	ret = True
	while (ret):
		timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
		print('Frame: ' + str(timestamp) + ' loading')
		ret,frame = cap.read()
		frame = frame.astype('uint8')

		# if the frame exists
		if ret:

			for obs in observers:
				gaze = datas[obs]

				# average the nearest gaze positions
				avgNow = averageGaze(gaze,timestamp,gmin,gmax)
				# now get the average gaze from 200 ms in the future
				avgFuture = averageGaze(gaze,timestamp+saccadeTime,gmin,gmax)
				
				dist = np.hypot(avgNow[0]-avgFuture[0],avgNow[1]-avgFuture[1])
				# if the average gaze jumps far enough
				if dist>minJump:
					print('Frame distance ' + np.str(dist) + ' exceeds jump of ' + str(minJump))
					# get the gaze map from 200 ms in the future
					gMap = gazeMap(gaze,cropSizes[1],cropSizes[1],timestamp,timestamp+saccadeTime,gmin,gmax,False)
					imsave('data/out/gaze_'+video+'_'+obs+'_'+str(timestamp)+'_200.tiff',gMap*255)
					# save the gaze cropped original images and the gaze map from 200 ms in the future
					for cropSize in cropSizes:
						crop = getCrop(frame,avgNow,cropSize)
						imsave('data/out/gaze_'+video+'_'+str(timestamp)+'_'+str(cropSize)+'.tiff',crop)






