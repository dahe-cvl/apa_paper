import sys
import os
import cv2,csv
import numpy as np
import matplotlib.pyplot as plt

from faceDetector import FaceDetection

class ImageExtractor:
	#members
	videofile = 0;
	frames = [];
	labels = [];
	vids = [];
	videonames = [];
	labellist = [];
	vid = 0;


	framecount = 0;
	fps = 0;
	duration = 0;
	framecountNew = 0;
	step = 0;

	faceDetector = 0;
		
	def __init__(self, input_path):
		print("create imageExtractor instance...");
		#print("create face and landmark detector...");
		self.faceDetector = FaceDetection(input_path + "shape_predictor_68_face_landmarks.dat");
		self.labellist = self.loadLabels(input_path + "training_gt.csv");
		

	def loadVideo(self, vfile):
		self.videofile = cv2.VideoCapture(vfile);
		ret = self.videofile.isOpened();
			
		#self.videonames = [];
		#self.vids = [];
		#self.labels = [];
		#self.frames = [];
		return ret;

	def extractNFrames(self, nImages, videoname):
		ret = self.videofile.isOpened();
		if (ret == True):
			self.framecount = self.videofile.get(cv2.CAP_PROP_FRAME_COUNT);	
			print("framecount: " + str(self.framecount));

			# get n frames of video
			step = int( self.framecount / (nImages + 1) );
			print("step: " + str(step));			

			if(step == 0):
				step = 1;
				print("WARNING: selected number of frames higher as framecount of video!");	

			startpos = step;
			cnt = 0;

			self.vid = self.vid + 1;
			print("vid: " + str(self.vid));
			label = self.searchLabels(videoname, self.labellist);

			for i in range(startpos, int(self.framecount)+1, int(step)):
				cnt = cnt + 1;
				ret,frame = self.videofile.read();
				if(ret <> False):
					if(self.faceDetector.findFace(frame) == True): 
						frame_crop = self.faceDetector.run(frame);
						self.frames.append(frame_crop);
						self.vids.append(self.vid);
						self.videonames.append(videoname);
						self.labels.append(label);

				if ( (cnt >= nImages) ):
					break;
			#print("shape of frames: " + str(self.frames[0].shape));
			#return self.frames;

	def reduceFrameRate(self, videoname, fpsNew):
		ret = self.videofile.isOpened();
		if (ret == True):

			self.framecount = self.videofile.get(cv2.CAP_PROP_FRAME_COUNT);	
			self.fps = int(self.videofile.get(cv2.CAP_PROP_FPS));
			self.duration = int(self.framecount / self.fps);

			self.framecountNew = self.duration * fpsNew;
			self.step = int( self.framecount / self.framecountNew );

			print("----------------------------");
			print("framecount original: " + str(self.framecount));
			print("fps original: " + str(self.fps));
			print("duration: " + str(self.duration));
			print("----------------------------");
			print("framecount new: " + str(self.framecountNew));
			print("fps new: " + str(fpsNew));
			print("duration: " + str(self.duration));
			print("step: " + str(self.step));	
			print("----------------------------");		


			if(self.step == 0):
				self.step = 1;
				print("WARNING: selected framerate is higher as original!");	

			startpos = self.step;
			cnt = 0;

			self.vid = self.vid + 1;
			print("vid: " + str(self.vid));
			label = self.searchLabels(videoname, self.labellist);

			for i in range(0, int(self.framecount), int(self.step)):
				cnt = cnt + 1;
				print(cnt);
				ret,frame = self.videofile.read();
				if(ret <> False):
					if(self.faceDetector.findFace(frame) == True): 
						frame_crop = self.faceDetector.run(frame);
						self.frames.append(frame_crop);
						self.vids.append(self.vid);
						self.videonames.append(videoname);
						self.labels.append(label);
	
				if ( (cnt >= self.framecountNew) ):
					break;
			#print("shape of frames: " + str(self.frames[0].shape));
			#return self.frames;

	def getVideonames(self):
		return self.videonames;

	def getVids(self):
		return self.vids;

	def getLabels(self):
		return self.labels;

	def getFrames(self):
		return self.frames;


	def loadLabels(self, csvfile):
		print("Load labels...");
		reader = csv.reader(open(csvfile,"rb"),delimiter=',');
		return list(reader);

	def searchLabels(self, name, labellist):
		searchCol =  [];
		for i in range(1,len(labellist)):
			searchCol.append(labellist[i][0]);
		index = searchCol.index(name) + 1;			
		return labellist[index][1:];	
		
	
	def playExtractedFrames(self):
		print("play video");
		cnt = 0;

		im = plt.imshow(self.frames[0]);
		print("print frame 0");
		for i in range(1, len(self.frames), 1):
			print("print frame " + str(i));
			im.set_data(self.frames[i]);
			plt.pause(0.02);
		plt.show();


	def printVideoDetails(self):
		print("\nDetails of video");
		print("---------------------");
		print("number of extracted frames: " + str(np.array(self.frames).shape[0]) );
		print("height of frames: " + str(np.array(self.frames).shape[1]) );
		print("width of frames: " + str(np.array(self.frames).shape[2]) );
		print("channels of frames: " + str(np.array(self.frames).shape[3]) );
		print("framecount: " + str(self.framecount));
		print("---------------------\n");	

	def playVideo(self):
		while(True):
			# Capture frame-by-frame
			ret, frame = self.videofile.read();

			if(ret <> False):
				cv2.imshow('frame', frame);
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break;

		# When everything done, release the capture
		self.videofile.release();
		cv2.destroyAllWindows();


