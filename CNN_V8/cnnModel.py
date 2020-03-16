from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,ZeroPadding2D
from keras.layers import Dropout, Activation
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2, activity_l2
from keras import backend as K
import time
import sys, csv, math, random, os
import numpy as np

from FaceDataset import FaceDataset
from MiniBatchGenerator import MiniBatchGenerator
from multiprocessing import Process, Queue


class cnnModel:
	BatchGeneratorTrain = None;
	BatchGeneratorVal = None;
	trainingDataset = None;
	validationDataset = None;
	#databaseName = "database_normalized.h5";
	databaseName = "database_normalized_25112016.h5";
	nEpochs = 200;
	batchsize = 32;
	loss = 0;
	train_acc = 0;
	val_acc = 0;
	output_path = "";
	acc_hist = [];
	max_acc_last = 0;
	
	# constructor
	def __init__(self, path, output_path):
		print("create instance of model");
		self.output_path = output_path;

		# load dataset
		print("load dataset...");
		self.trainingDataset = FaceDataset(path, self.databaseName, "train");
		self.validationDataset = FaceDataset(path, self.databaseName, "val");

		trainingDatasetSamples = self.trainingDataset.size();
		print("trainingset sample size: " + str(trainingDatasetSamples));

		validationDatasetSamples = self.validationDataset.size();
		print("validationset sample size: " + str(validationDatasetSamples));

		self.BatchGeneratorTrain = MiniBatchGenerator(self.trainingDataset, self.batchsize);
		self.BatchGeneratorVal = MiniBatchGenerator(self.validationDataset, self.batchsize);		
		
	def run(self, option):		
		# create model
		model = self.createModel(0, 0, 0);
		#self.findBestHyperparameter(10);
		# train model
		if(option == 1):
			self.trainModel(model, 0);

	def createModel(self, lRate, decay, wDecay):
		model = Sequential();

		model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(64, 64, 3)));
		model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',));
		model.add(MaxPooling2D((2,2), strides=(2,2)));

		model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'));
		model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'));
		model.add(MaxPooling2D((2,2), strides=(2,2)));

		model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'));
		model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'));
		model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'));
		model.add(MaxPooling2D((2,2), strides=(2,2)));

		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'));
		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'));
		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'));
		model.add(MaxPooling2D((2,2), strides=(2,2)));

		model.add(Flatten());
		model.add(Dense(4096, activation='relu')); # , W_constraint = maxnorm(2), W_regularizer=l2(wDecay), W_regularizer=l2(wDecay)
		model.add(Dropout(0.5));
		model.add(Dense(4096, activation='relu'));
		model.add(Dropout(0.5));
		model.add(Dense(5, activation='softmax'));

		#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0);
		#sgd = SGD(lr=lRate, decay=decay, momentum=0.9, nesterov=True);
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']);	
		
		print(model.summary());
		#print("Parameters --> Learning rate: " + str(lRate) + ", Decay: " + str(decay) + ", WDecay (FC): " + str(wDecay));
		return model;
	
	def findBestHyperparameter(self, nCombinations):
		lRate = 0;
		wDecay = 0;
		
		# test X combinations
		for i in range(0, nCombinations):
			self.run_in_separate_process(self.trainModelNew, (i, ));
			#wDecay = self.randomWeightDecay([-6, -1]);
			#lRate = self.randomLearningRate([-6, -1]);
			#wDecay = 0.5;
			#decay = 0.00001;
			#lRate = 0.1;			
			#model = self.createModel(lRate, decay, wDecay);
			#print("[" + str(i) + "]: learning rate: " + str(lRate) + ", decay: " + str(decay)+ ", weight decay: " + str(wDecay));
			#self.trainModel(model, i);
	
	def trainModelNew(self, i):
		
		wDecay = self.randomWeightDecay([-6, -1]);
		lRate = self.randomLearningRate([-4, -1]);
		#wDecay = 0.5;
		decay = 0.00001;
		lRate = 0.001;			
		model = self.createModel(lRate, decay, wDecay);
		print('model training\'s PID:', os.getpid());
		print("[" + str(i) + "]: learning rate: " + str(lRate) + ", decay: " + str(decay)+ ", weight decay: " + str(wDecay));
		self.trainModel(model, i);

		
	def run_in_separate_process(self, method, args):
		def queue_wrapper(q, params):
			r = method(*params)
			q.put(r)

		q = Queue()
		p = Process(target=queue_wrapper, args=(q, args))
		p.start()
		#return_val = q.get()
		p.join()
		#return return_val		
			

	def trainModel(self, model, i):
		#training and evaluation
		early_stopping = EarlyStopping(monitor='val_acc', patience=10);
		#lrate = LearningRateScheduler(self.step_decay);
		bestModel = ModelCheckpoint(self.output_path +"model.{epoch:02d}-{val_acc:.4f}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto');
		#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.4, verbose=1, patience=5, min_lr=0.000001)
		callbacks_list = [early_stopping, bestModel];

		history_training = model.fit_generator(self.BatchGeneratorTrain.BatchGenerator(), samples_per_epoch=self.trainingDataset.size(), nb_epoch=self.nEpochs,  max_q_size=1, pickle_safe=False, verbose=1, callbacks=callbacks_list, validation_data=self.BatchGeneratorVal.BatchGenerator(), nb_val_samples=self.validationDataset.size());	
					
		self.saveTrainingHistoryToCSV(history_training.history['loss'], history_training.history['acc'], history_training.history['val_loss'], history_training.history['val_acc']);
		self.saveModelWeights(model, self.output_path + "weights.h5");
		self.saveModel(model, self.output_path + "model_final.h5");
	
	def randomWeightDecay(self, wRange):
		value = 10 ** round(random.uniform(wRange[0],wRange[1]), 0);
		return value;

	def randomLearningRate(self, lRange):
		#value = round(random.uniform(lRange[0], lRange[1]), digits);
		value = 10 ** round(random.uniform(lRange[0],lRange[1]), 0);
		return value;

	def initWeights(self, n):	
		w = np.random.randn(n) * sqrt(2.0/n);

	def step_decay(self, epoch):
		#initial_lrate = 0.01;
		#drop = 0.5;
		#epochs_drop = 5.0;
		#lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop));
		lrate = 10 ** random.uniform(-6, 1);
		print("learning rate: " + str(lrate));
		return lrate;

	def saveTrainingHistoryToCSV(self, loss, acc, val_loss, val_acc):
		#save training history to csv file
		metricsFile = open(self.output_path + "metrics.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(loss, acc, val_loss, val_acc);
		csvWriter.writerows(csvdata);
		metricsFile.close();

	def calculatePredictions(self, model, data):
		# calculate predictions
		predictions = model.predict(data);
		return predictions;	
		
	def saveModelWeights(self, model, filename):
		# save trained model - weights
		model.save_weights(filename);

	def saveModel(self, model, filename):
		# save trained model
		model.save(filename);
		
	def loadModelWeights(self, model, filename):
		model.load_weights(filename);

	def printDetails(self):
		print(model.history);

	
	


