import numpy as np
import matplotlib.pyplot as plt
from random import randint
from DataAugmentation import DataAugmentation

class MiniBatchGenerator:
	# Create minibatches of a given size from a dataset.
	# Preserves the original sample order unless shuffle() is used.
	nSeq = 0;
	batchsize = 0;
	dataset = None;
	tform = None; 
	stat = None;
	nBatches = 0;
	b = [];
	dataGenerator = None;
	mode = 0;
	shuffled_idx = [];

	def __init__(self, dataset, split, nSeq, bs):
		# Constructor.
		# dataset is a ClassificationDataset to wrap.
		# bs is an integer specifying the minibatch size.
		# tform is an optional SampleTransformation.
		# If given, tform is applied to all samples returned in minibatches.
		self.dataset = dataset;
		self.batchsize = bs;
		self.nSeq = nSeq;

		if((self.dataset.getNClasses() % self.batchsize) == 0 ):
			self.nBatches = int(self.dataset.getNClasses() / self.batchsize);
		else:
			self.nBatches = int(self.dataset.getNClasses() / self.batchsize)+1;

		print("nBatches: " + str(self.nBatches));
		print("nSequences: " + str(self.nSeq));

		self.dataGenerator = DataAugmentation();

		if(split == "train"):
			self.mode = 0;
		elif(split == "val"):
			self.mode = 1;
		elif(split == "test"):
			self.mode = 2;

	def batchsize(self):
		# Return the number of samples per minibatch.
		# The size of the last batch might be smaller.
		return self.batchsize;

	def nbatches(self):
		# Return the number of minibatches.
		return self.nBatches;

	def shuffle(self):
		# Shuffle the dataset samples so that each
		# ends up at a random location in a random minibatch.
		return 0;

	def batch(self, bid):
		# Return the bid-th minibatch.
		# Batch IDs start with 0 and are consecutive.
		# Throws an error if the minibatch does not exist.
		samples = [];
		labels = [];
		if(self.mode == 0 or self.mode == 1):
			if( bid < self.nBatches and bid >= 0):
				#for i in range(0, self.batchsize):
				i = 0;
				vid = 1;
				while(i < self.batchsize):
					offset = bid * self.batchsize;
					if((offset+i) >= self.dataset.getNClasses() ):
						break;
					#print(str(offset) + " + " + str(i) + " = " + str(offset + i) + " --> " + str(offset + vid));

					ids, n, s, l = self.dataset.getAllSamplesOfID( offset + vid );
					#print("ids shape: " + str(ids.shape));
					if(ids.shape[0] <> 0):
						s_n = np.zeros((self.nSeq, s.shape[1], s.shape[2], s.shape[3]));
						l_n = np.zeros((self.nSeq, l.shape[1]));
						if(self.nSeq > s.shape[0]):
							s_n[:s.shape[0],:,:,:] = s[:s.shape[0],:,:,:];
							l_n[:s.shape[0],:] = l[:s.shape[0],:];
						elif(self.nSeq <= s.shape[0]):
							s_n[:self.nSeq,:,:,:] = s[:self.nSeq,:,:,:];
							l_n[:self.nSeq,:] = l[:self.nSeq,:];
						
						tmp_b = [s_n, l_n];
						#self.dataGenerator.showImages(tmp_b);
						b_dataAug = self.dataGenerator.runDataAugmentation(tmp_b);
						#self.dataGenerator.showImages(b_dataAug);
						samples.append(b_dataAug[0]);
						labels.append(l_n[0]);

						i = i + 1;
						vid = vid + 1;
					else:
						vid = vid + 1;
				samples = np.array(samples);
				labels = np.array(labels);
				self.b = (samples, labels);
				
				return self.b;
			else:
				print("ERROR: index out of range!");
				exit();
		elif(self.mode == 2):
			if( bid < self.nBatches and bid >= 0):
				#for i in range(0, self.batchsize):
				i = 0;
				vid = 1;
				while(i < self.batchsize):
					offset = bid * self.batchsize;
					if((offset+i) >= self.dataset.getNClasses() ):
						break;
					#print(str(offset) + " + " + str(i) + " = " + str(offset + i) + " --> " + str(offset + vid));

					ids, n, s = self.dataset.getAllTestSamplesOfID( offset + vid );
					#print("ids shape: " + str(ids.shape));
					if(ids.shape[0] <> 0):
						s_n = np.zeros((self.nSeq, s.shape[1], s.shape[2], s.shape[3]));
						if(self.nSeq > s.shape[0]):
							s_n[:s.shape[0],:,:,:] = s[:s.shape[0],:,:,:];
						elif(self.nSeq <= s.shape[0]):
							s_n[:self.nSeq,:,:,:] = s[:self.nSeq,:,:,:];
						
						samples.append(s_n);

						i = i + 1;
						vid = vid + 1;
					else:
						vid = vid + 1;
				samples = np.array(samples);
				self.b = (samples);
				
				
				return self.b;
			else:
				print("ERROR: index out of range!");
				exit();

	def batch_Random(self, bid, r_idx):
		# Return the bid-th minibatch.
		# Batch IDs start with 0 and are consecutive.
		# Throws an error if the minibatch does not exist.
		samples = [];
		labels = [];
		if(self.mode == 0 or self.mode == 1):
			if( bid < self.nBatches and bid >= 0):
				#for i in range(0, self.batchsize):
				i = 0;
				vid = 1;
				while(i < self.batchsize):
					offset = bid * self.batchsize;
					if((offset+i) >= self.dataset.getNClasses() ):
						break;
					#print(str(offset) + " + " + str(i) + " = " + str(offset + i) + " --> " + str(offset + vid));

					ids, n, s, l = self.dataset.getAllSamplesOfID( offset + vid );

					#print("ids shape: " + str(ids.shape));
					if(ids.shape[0] <> 0):	

						if(ids.shape[0] == 60):
							

							# take samples with generated index
							r_s = s[r_idx];
							r_l = l[r_idx];

							#print("random samples: " + str(r_shape));
							#print("random labels: " + str(r_l.shape));

							tmp_b = [r_s, r_l];
						else:
							s_n = np.zeros((10, s.shape[1], s.shape[2], s.shape[3]));
							l_n = np.zeros((10, l.shape[1]));
							if(10 > s.shape[0]):
								s_n[:s.shape[0],:,:,:] = s[:s.shape[0],:,:,:];
								l_n[:s.shape[0],:] = l[:s.shape[0],:];
							elif(10 <= s.shape[0]):
								s_n[:10,:,:,:] = s[:10,:,:,:];
								l_n[:10,:] = l[:10,:];
						
							tmp_b = [s_n, l_n];

						#self.dataGenerator.showImages(tmp_b);
						#b_dataAug = self.dataGenerator.runDataAugmentation(tmp_b);
						#self.dataGenerator.showImages(b_dataAug);
						samples.append(tmp_b[0]);
						labels.append(tmp_b[1][0]);

						i = i + 1;
						vid = vid + 1;
					else:
						vid = vid + 1;

				samples = np.array(samples);
				labels = np.array(labels);
				self.b = (samples, labels);
				#print("------------------------------------");
				#print("bid: " + str(bid) + " - " + str(samples.shape));
				
				return self.b;
			else:
				print("ERROR: index out of range!");
				exit();
		elif(self.mode == 2):
			if( bid < self.nBatches and bid >= 0):
				for i in range(0, self.batchsize, 1):
					offset = bid * self.batchsize;
					if((offset+i) >= self.dataset.size() ):
						break;
					s = self.dataset.sample(offset + i);
					samples.append(s);
				samples = np.array(samples);
				self.b = (samples);
				return self.b;
			else:
				print("ERROR: index out of range!");
				exit();


	def batchNew(self, bid):
		# Return the bid-th minibatch.
		# Batch IDs start with 0 and are consecutive.
		# Throws an error if the minibatch does not exist.
		samples = [];
		labels = [];
		if( bid < self.nBatches and bid >= 0):
			for i in range(0, self.batchsize):
				offset = bid * self.batchsize;
				if((offset+i) >= self.dataset.size() ):
					break;
				s, l = self.dataset.sample(offset + i);
				samples.append(s);
				labels.append(l);
			samples = np.array(samples);
			labels = np.array(labels);
			self.b = (samples, labels);
			return self.b;
		else:
			print("ERROR: index out of range!");
			exit();

	def BatchGenerator(self):
		if(self.mode == 0):
			
			
			# training mode
			while(1):
				#print("\nshuffle batch idxs");
				self.shuffled_idx = np.arange(self.nbatches());
				np.random.shuffle(self.shuffled_idx);

				#print("\ngenerate new random indexes ...");
				# generate random number
				r_idx = [];
				for a in range(0, 10):
					start = a*6;
					end = (a*6 + 6)-1;
					r_tmp = randint(start, end);
					r_idx.append(r_tmp);

				#print("random numbers: " + str(r_idx));

				#for a in range(0, self.nbatches()):
				for a in range(0, len(self.shuffled_idx)):
					#print("\nbatch: " + str(a));
					#batch = self.batch(a);		
					#batch = self.batch(self.shuffled_idx[a]);
					batch = self.batch_Random(self.shuffled_idx[a], r_idx);		
					#fBatch, lBatch = self.dataGenerator.runDataAugmentation(batch);
					#self.dataGenerator.showImages(batch);
					#print(str(a) + ": " + str(batch[0].shape));
					#yield fBatch, lBatch;
					#print(batch[0]);
					#print(batch[1]);
					yield batch[0], batch[1];
		elif(self.mode == 1):
			# validation mode
			while(1):
				for a in range(0, self.nbatches()):
					batch = self.batch(a);		
					#fBatch, lBatch = self.dataGenerator.runDataAugmentation(batch);
					#print(str(a) + ": " + str(fBatch.shape));
					#yield fBatch, lBatch;
					yield batch[0], batch[1];
		elif(self.mode == 2):
			# test mode
			while(1):
				for a in range(0, self.nbatches()):
					batch = self.batch(a);		
					#fBatch, lBatch = self.dataGenerator.runDataAugmentation(batch);
					#print(str(a) + ": " + str(batch.shape));
					#yield fBatch, lBatch;
					yield batch;


	def printBatchImages(self, b):
		for i in range(0, int(b.shape[0]) , 1):
			im = plt.imshow(b[i]);
			plt.pause(0.6);
		plt.show();

