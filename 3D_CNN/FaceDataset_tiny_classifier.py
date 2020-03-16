from ClassificationDataset import ClassificationDataset
import numpy as np
import h5py as hf
import cv2
import matplotlib.pyplot as plt
from keras.utils import np_utils

class FaceDataset_tiny_classifier(ClassificationDataset):
	# The Face dataset.
	nSamples = 0;
	nSamplesTrain = 0;
	nSamplesVal = 0;
	nSamplesTest = 0;
	completeSamples = 0;
	nClasses = 0;
	path = "";
	split = "";
	database = None;
	labels = [];

	def __init__(self, fdir, databaseName, split):
		# Ctor. fdir is a path to a directory in which the Face.h5
		# files reside (e.g. data_batch_1 and test_batch).
		# split is a string that specifies which dataset split to load.
		# Can be 'train' (training set), 'val' (validation set) or 'test' (test set).
		completeSamples = 0;		
		try:
			print(fdir);
			self.path = fdir;
			self.database = hf.File(self.path + databaseName, 'r');	 # databaseName = "database_normalized_NEW.h5"

			ids = np.array(self.database["id"][:]);
			self.completeSamples = ids.shape[0];		

		except:
			print("ERROR: Cannot open dataset!");
			exit();
		print(split);
		if (split <> "train" and split <> "val" and split <> "test"):
			print("ERROR: false split option selected!");
			self.database.close();
			exit();
		self.split = split;


		# sort dataset 
		sortedIds_Idx = np.argsort(ids);
		sorted_Ids = ids[sortedIds_Idx];

		# find range
		sids = np.array(sorted_Ids);
		condition = (sids==4801);
		foundIdx = np.array(np.where(condition));
		splitIdx = foundIdx[0][0];
		print(splitIdx);
		
		if(self.split == "train"):
			self.nSamples = 5000;
			#self.nSamples = round(self.completeSamples * 0.8);
		elif(self.split == "val"):
			self.nSamples =  1000;
			#self.nSamples =  round(self.completeSamples * 0.2);
		elif(self.split == "test"):
			self.nSamples = nSamples;

		# create random labels
		self.labels = np.random.randint(5, size=(self.completeSamples,));
		print(self.labels);
		self.labels = np_utils.to_categorical(self.labels, 5);
		print(self.labels);

		if(self.nSamples == 0):
			print("ERROR: sample size is zero!");
			self.database.close();
			exit();
		
	# implement the other members ...
	def size(self):
		# Returns the size of the dataset (number of images).
		return self.nSamples;


	def nclasses(self):
		# Returns the number of different classes.
		# Class labels start with 0 and are consecutive.
		return self.nClasses;


	def classname(self, cid):
		# Returns the name of a class as a string.
		return 0;
		
	def sample(self, sid):
		# Returns the sid-th sample in the dataset, and the
		# corresponding class label. Depending of your language,
		# this can be a Matlab struct, Python tuple or dict, etc.
		# Sample IDs start with 0 and are consecutive.
		# The channel order of samples must be RGB.
		# Throws an error if the sample does not exist.

		if(self.split == "train" and sid < self.nSamples and sid >= 0):	
			sample = np.array(self.database["data"][sid]);
			b,g,r = cv2.split(sample);
			rgb_frame = cv2.merge([r,g,b]);
			labels = self.labels[sid];
			return sample, labels;
		elif(self.split == "val" and sid < self.nSamples and sid >= 0):
			valOffset = (self.completeSamples - self.nSamples);
			index = valOffset + sid;	
			sample = np.array(self.database["data"][index]);
			b,g,r = cv2.split(sample);
			rgb_frame = cv2.merge([r,g,b]);
			labels = self.labels[index];
			return sample, labels;
		else:
			print("ERROR: index out of range!");
			self.database.close();
			exit();
	def getVID(self, sid):
		vid = self.database["id"][sid];
		return vid;

	def getAllSamplesOfID(self, vid):
		
		samplesOfID = [];
		samples = [];
		sids = np.array(self.database["id"][:]);
		condition = (sids==vid);
		foundIdx = np.array(np.where(condition));
		for i in range(0, foundIdx.shape[1], 1):
			samples.append(self.database["data"][foundIdx[0][i]]);
		samples = np.array(samples);
		return np.array(samples);
	
	def printSamples(self, samples):
		for i in range(0, int(samples.shape[0]) , 1):
			im = plt.imshow(samples[i]);
			plt.pause(0.6);
		plt.show();

	def getTrainData(self):
		return np.array(self.database["data"][:]);