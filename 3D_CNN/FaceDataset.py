from ClassificationDataset import ClassificationDataset
import numpy as np
import h5py as hf
import cv2
import matplotlib.pyplot as plt
from DataAugmentation import DataAugmentation

class FaceDataset(ClassificationDataset):
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
	dataGenerator = None;

	def __init__(self, fdir, databaseName, split):
		# Ctor. fdir is a path to a directory in which the Face.h5
		# files reside (e.g. data_batch_1 and test_batch).
		# split is a string that specifies which dataset split to load.
		# Can be 'train' (training set), 'val' (validation set) or 'test' (test set).
		completeSamples = 0;		
		try:
			print(str(fdir) + "/" + databaseName);
			self.path = fdir;
			self.database = hf.File(self.path + databaseName, 'r');	 # databaseName = "database_normalized_NEW.h5"
			
		except:
			print("ERROR: Cannot open dataset!");
			exit();

		ids = np.array(self.database["id"][:]);
		self.completeSamples = ids.shape[0];
		self.nClasses = np.unique(np.sort(ids)).shape[0];

		print("nSamles: " + str(self.completeSamples));
		print("nClasses: " + str(self.nClasses));



		print(split);
		if (split <> "train" and split <> "val" and split <> "test"):
			print("ERROR: false split option selected!");
			self.database.close();
			exit();
		self.split = split;
		
		if(self.split == "train"):
			self.nSamples = self.completeSamples;
		elif(self.split == "val"):
			self.nSamples =  self.completeSamples;
		elif(self.split == "test"):
			self.nSamples = self.completeSamples;
			self.dataGenerator = DataAugmentation();


		if(self.nSamples == 0):
			print("ERROR: sample size is zero!");
			self.database.close();
			exit();
		
	# implement the other members ...
	def size(self):
		# Returns the size of the dataset (number of images).
		return self.nSamples;


	def getNClasses(self):
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
			# zero mean - normalization
			sample = sample.astype('float32');
			b,g,r = cv2.split(sample);

			## FACE Dataset
			r = (r - 101.297884206) / 55.4421940732;
			g = (g - 109.858226172) / 56.8107097886;
			b = (b - 147.083965371) / 64.9683332671;

			rgb_frame = cv2.merge([r,g,b]);
			labels = np.array(self.database["labels"][sid]);

			return rgb_frame, labels;
		elif(self.split == "val" and sid < self.nSamples and sid >= 0):
			sample = np.array(self.database["data"][sid]);
			# zero mean - normalization
			sample = sample.astype('float32');
			b,g,r = cv2.split(sample);

			## FACE Dataset
			r = (r - 101.297884206) / 55.4421940732;
			g = (g - 109.858226172) / 56.8107097886;
			b = (b - 147.083965371) / 64.9683332671;

			rgb_frame = cv2.merge([r,g,b]);
			labels = np.array(self.database["labels"][sid]);

			return rgb_frame, labels;
		elif(self.split == "test" and sid < self.nSamples and sid >= 0):
			sample = np.array(self.database["data"][sid]);
			
			# zero mean - normalization
			sample = sample.astype('float32');
			b,g,r = cv2.split(sample);

			## BODY dataset
			#r = (r - 118.103158522) / 74.2845685811;
			#g = (g - 97.8464310079) / 68.3421720092;
			#b = (b - 93.9819708838) / 66.5148472604;

			## FACE Dataset
			r = (r - 101.297884206) / 55.4421940732;
			g = (g - 109.858226172) / 56.8107097886;
			b = (b - 147.083965371) / 64.9683332671;

			frame_norm = cv2.merge([r,g,b]);

			# ten-crop oversampling
			samples = self.dataGenerator.runOverSampling(frame_norm);

			return samples;
		else:
			print("ERROR: index out of range!");
			self.database.close();
			exit();

	def getVID(self, sid):
		if(self.split == "train" and sid < self.nSamples and sid >= 0):	
			vid = self.database["id"][sid];
			return vid;
		elif(self.split == "val" and sid < self.nSamples and sid >= 0):
			vid = self.database["id"][sid];
			return vid;
		elif(self.split == "test" and sid < self.nSamples and sid >= 0):
			vid = self.database["id"][sid];
			return vid;
		else:
			print("ERROR: index out of range!");
			self.database.close();
			exit();
		return vid;

	def getName(self, sid):
		if(self.split == "train" and sid < self.nSamples and sid >= 0):	
			name = self.database["name"][sid];
			return name;
		elif(self.split == "val" and sid < self.nSamples and sid >= 0):
			name = self.database["name"][sid];
			return name;
		elif(self.split == "test" and sid < self.nSamples and sid >= 0):
			name = self.database["name"][sid];
			return name;
		else:
			print("ERROR: index out of range!");
			self.database.close();
			exit();
		return "";
	
	def getAllSamplesOfID(self, vid):
		
		samplesOfID = [];
		samples = [];
		labels = [];
		names = [];
		ids = [];
		sids = np.array(self.database["id"][:]);
		condition = (sids==vid);
		foundIdx = np.array(np.where(condition));
		for i in range(0, foundIdx.shape[1], 1):
			samples.append(self.database["data"][foundIdx[0][i]]);
			labels.append(self.database["labels"][foundIdx[0][i]]);
			names.append(self.database["name"][foundIdx[0][i]]);
			ids.append(self.database["id"][foundIdx[0][i]]);
		samples = np.array(samples);
		labels = np.array(labels);
		names = np.array(names);
		ids = np.array(ids);
		
		# apply statistics
		#norm_samples = self.applyStatistics(samples);
		norm_samples = samples;
		return ids, names, norm_samples, labels;

	def getAllTestSamplesOfID(self, vid):
		
		samplesOfID = [];
		samples = [];
		names = [];
		ids = [];
		sids = np.array(self.database["id"][:]);
		condition = (sids==vid);
		foundIdx = np.array(np.where(condition));
		for i in range(0, foundIdx.shape[1], 1):
			samples.append(self.database["data"][foundIdx[0][i]]);
			names.append(self.database["name"][foundIdx[0][i]]);
			ids.append(self.database["id"][foundIdx[0][i]]);
		samples = np.array(samples);
		names = np.array(names);
		ids = np.array(ids);
		
		# apply statistics
		norm_samples = self.applyStatistics(samples);

		return ids, names, norm_samples;
	
	
	def applyStatistics(self, samples):
		new_samples = [];
		for i in range(0, samples.shape[0]):
			b,g,r = cv2.split(samples[i]);
			## FACE Dataset
			r = (r - 101.297884206) / 55.4421940732;
			g = (g - 109.858226172) / 56.8107097886;
			b = (b - 147.083965371) / 64.9683332671;
			rgb_frame = cv2.merge([r,g,b]);	
			new_samples.append(rgb_frame);
		new_samples = np.array(new_samples);
		return new_samples;

	def printSamples(self, samples):
		for i in range(0, int(samples.shape[0]) , 1):
			im = plt.imshow(samples[i]);
			plt.pause(0.6);
		plt.show();

	def getTrainData(self):
		return np.array(self.database["data"][:]);
