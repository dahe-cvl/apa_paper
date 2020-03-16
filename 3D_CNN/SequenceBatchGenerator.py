import numpy as np
import matplotlib.pyplot as plt
from DataAugmentation import DataAugmentation

class SequenceBatchGenerator:
	# Create minibatches of a given size from a dataset.
	# Preserves the original sample order unless shuffle() is used.

	batchsize = 0;
	dataset = None;
	tform = None; 
	stat = None;
	nBatches = 0;
	b = [];
	dataGenerator = None;
	mode = 0;
	shuffled_idx = [];
	sequences = 6;

	def __init__(self, dataset, split, sequences):
		# Constructor.
		# dataset is a ClassificationDataset to wrap.
		# bs is an integer specifying the minibatch size.
		# tform is an optional SampleTransformation.
		# If given, tform is applied to all samples returned in minibatches.
		self.dataset = dataset;

		self.sequences = sequences;
		print(self.sequences);
		self.dataGenerator = DataAugmentation();

		if(split == "train"):
			self.mode = 0;
		elif(split == "val"):
			self.mode = 1;
		elif(split == "test"):
			self.mode = 2;

	def SequenceGenerator(self):
		if(self.mode == 'train'):
			while(1):
				for i in range(1, 4801):
					# get samples of VID
					ids, names, samples, labels = getAllSamplesOfID(vid);
				
					s = np.zeros((self.sequences, samples.shape[1], samples.shape[2], samples.shape[3]));

					if(self.sequences > samples.shape[0]):
						s[:samples.shape[0],:,:,:] = train_x[:samples.shape[0],:,:,:];
					elif(self.sequences <= samples.shape[0]):
						s[:self.sequences,:,:,:] = train_x[:self.sequences,:,:,:];

					s = np.reshape(s, (1, self.sequences, samples.shape[1], samples.shape[2], samples.shape[3]));
					l = labels[:1, :];

					yield s, l;

		elif(self.mode == 'val'):
			while(1):
				for i in range(4800, 6001):
					# get samples of VID
					ids, names, samples, labels = getAllSamplesOfID(vid);
				
					s = np.zeros((self.sequences, samples.shape[1], samples.shape[2], samples.shape[3]));

					if(self.sequences > samples.shape[0]):
						s[:samples.shape[0],:,:,:] = train_x[:samples.shape[0],:,:,:];
					elif(self.sequences <= samples.shape[0]):
						s[:self.sequences,:,:,:] = train_x[:self.sequences,:,:,:];

					s = np.reshape(s, (1, self.sequences, samples.shape[1], samples.shape[2], samples.shape[3]));
					l = labels[:1, :];
				
					yield s, l;


	def printSequenceImages(self, b):
		for i in range(0, int(b.shape[0]) , 1):
			im = plt.imshow(b[i]);
			plt.pause(0.6);
		plt.show();

