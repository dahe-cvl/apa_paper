from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,ZeroPadding2D
from keras.layers import Dropout, Activation
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution2D,Convolution3D, UpSampling3D
from keras.layers.convolutional import MaxPooling2D, MaxPooling3D
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.models import load_model, model_from_json
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2, activity_l2
from keras import backend as K
import tensorflow as tf
from keras import initializations
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from MiniBatchGenerator import MiniBatchGenerator
import sys, csv, math, random, os, time,datetime
import numpy as np
import mock
from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
from random import randint
from matplotlib import pyplot

class Cnn3DModel:
	output_path = "";
	BatchGeneratorTrain = None;
	BatchGeneratorVal = None;
	trainingDataset = None;
	validationDataset = None;
	testDataset = None;
	model = None;
	trainingHistory = None;

	nEpochs = 300;
	batchsize = 16;
	sequences = 10;

	sample_shape = [];

	callbacks_list = [];

	# hyperparameter
	lRate = 0;
	lDecay = 0;
	beta_1 = 0;
	beta_2 = 0;
	epsilon = 0;

	wDecay = 0.002;
	pDropout = 0;

	optim = 'sgd';

	convout1 = None;
	results = "";
	
	def __init__(self, trainingDataset, validationDataset, testDataset, output_path):
		print("create instance");
		self.output_path = output_path;
		self.trainingDataset = trainingDataset;
		self.validationDataset = validationDataset;
		self.testDataset = testDataset;

		sample, labels = self.trainingDataset.sample(0);
		print(sample.shape[0]);
		print(sample.shape[1]);
		print(sample.shape[2]);
		self.sample_shape = np.zeros(4);
		self.sample_shape = self.sample_shape.astype('int32');
		self.sample_shape[0] = self.sequences;
		self.sample_shape[1] = sample.shape[0];
		self.sample_shape[2] = sample.shape[1];
		self.sample_shape[3] = sample.shape[2];
		
		self.BatchGeneratorTrain = MiniBatchGenerator(self.trainingDataset, 'train', self.sequences, self.batchsize);
		self.BatchGeneratorVal = MiniBatchGenerator(self.validationDataset, 'val', self.sequences, self.batchsize);	
		self.BatchGeneratorTest = MiniBatchGenerator(self.testDataset, 'test', self.sequences, self.batchsize);	
		
		initializations.initWeights = self.initWeights;

	def createModel(self):
		# create new result folder
		ts = time.time();
		
		self.results = "Results_" + str(datetime.datetime.fromtimestamp(ts).strftime('%d%m%Y_%H%M%S')) + "/";
    		os.mkdir(self.output_path + self.results, 0755 );

		print("create model...");
		batchMode = 2;		
		self.model = Sequential();
		
		# 1st layer group
		self.model.add(Convolution3D(64, 5, 5, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv1a', input_shape=(self.sample_shape[0], self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(64, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv1b'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool1'));

		# 2nd layer group
		self.model.add(Convolution3D(128, 5, 5, 2, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv2a'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(128, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv2b'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool2'));

		# 3rd layer group
		self.model.add(Convolution3D(256, 5, 5, 1, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv3a'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(256, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv3b'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(256, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv3c'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool3'));

		# 4rd layer group
		self.model.add(Convolution3D(512, 5, 5, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv4a'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(512, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv4b'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(512, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv4c'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool4'));
		
		self.model.add(Flatten());
		
		# FC layers group
		self.model.add(Dense(1024, W_regularizer=l2(self.wDecay), name='fc1'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		self.model.add(Dropout(self.pDropout));

		self.model.add(Dense(5, activation='sigmoid', name='fc2'));

	def addDecoder(self):
		
		print("add decoder part model...");
		batchMode = 2;		
		self.model = Sequential();
		
		# 1st layer group
		self.model.add(Convolution3D(64, 5, 5, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv1a', input_shape=(self.sample_shape[0], self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(64, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv1b'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool1'));

		# 2nd layer group
		self.model.add(Convolution3D(128, 5, 5, 2, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv2a'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(128, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv2b'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool2'));

		# 3rd layer group
		self.model.add(Convolution3D(256, 5, 5, 1, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv3a'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(256, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv3b'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(256, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv3c'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool3'));

		# 4rd layer group
		self.model.add(Convolution3D(512, 5, 5, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv4a'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(512, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv4b'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		#self.model.add(Convolution3D(512, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv4c'));
		#self.model.add(BatchNormalization(mode=batchMode));
		#self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool4'));
		
		self.model.add(Flatten());
		
		# FC layers group
		self.model.add(Dense(1024, W_regularizer=l2(self.wDecay), name='fc1'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		self.model.add(Dropout(self.pDropout));

		self.model.add(Dense(5, activation='sigmoid', name='fc2'));



		print("load best model weights...");
		self.loadWeights(self.output_path + "Model/weights_best.h5");

		#print(self.model.layers.get_weights());
		self.model.layers.pop()
		self.model.layers.pop()
		self.model.layers.pop()
		self.model.layers.pop()
		self.model.layers.pop()
		self.model.layers.pop()

		self.model.outputs = [self.model.layers[-1].output]
		self.model.layers[-1].outbound_nodes=[]		
		
		# decode
		self.model.add(UpSampling3D(size=(2, 2, 2) ))
		self.model.add(Convolution3D(64, 5, 5, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='deconv1a'));
		self.model.add(Activation('relu'));
		self.model.add(UpSampling3D(size=(1, 2, 2) ))
		self.model.add(Convolution3D(128, 5, 5, 2, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='deconv2a'));
		self.model.add(Activation('relu'));
		self.model.add(UpSampling3D(size=(2, 2, 2)))
		self.model.add(Convolution3D(256, 5, 5, 1, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='deconv3a'));
		self.model.add(Activation('relu'));
		self.model.add(UpSampling3D(size=(2, 2, 2) ))
		self.model.add(Convolution3D(512, 5, 5, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='deconv4a'));
		self.model.add(Activation('relu'));



	def compileModel(self):
		print("compile model...");
		self.model.compile(optimizer=self.optim, loss='binary_crossentropy');	#, metrics=['accuracy'], 

	def smooth_huber_loss(self, y_true, y_pred):
		    #"""Regression loss function, smooth version of Huber loss function. """
		W = 1.5;
		
		cosh = (K.exp(y_true - y_pred) + K.exp((-1) * (y_true - y_pred))) / 2;

		return K.mean(w * K.log(cosh));

	def mse_loss(self, y_true, y_pred):
		return K.mean(K.square(y_pred - y_true));

	def acc(self, y_true, y_pred):
		return K.mean(K.mean(K.abs(y_pred - y_true),axis=0));

	def huber_loss(self, y_true, y_pred):
		    #"""Regression loss function, Huber loss function. """
		c = 0.6;
		
		r = (y_true - y_pred);
		r_abs = K.abs(r);

		if K.lesser_equal(r_abs, c) is not None:
  			p = K.square(r); 
		elif K.greater(r_abs > c) is not None:
  			p = c * (2 * r_abs - c);


		return K.mean(p);

	def trainModel(self):
		print("train model...");

		self.trainingHistory = self.model.fit_generator(self.BatchGeneratorTrain.BatchGenerator(), 
							samples_per_epoch=self.trainingDataset.getNClasses(), 
							nb_epoch=self.nEpochs,  
							max_q_size=1, 
							pickle_safe=True, 
							verbose=1, 
							callbacks=self.callbacks_list, 
							validation_data=self.BatchGeneratorVal.BatchGenerator(), 
							nb_val_samples=self.validationDataset.getNClasses());	
		return self.trainingHistory;		


	def evaluateModel(self):
		print("evaluate model...");
		self.evaluationHistory = self.model.evaluate_generator(self.BatchGeneratorVal.BatchGenerator(), val_samples=self.validationDataset.getNClasses(), max_q_size=1, pickle_safe=True);
		#self.evaluationHistory = self.model.evaluate_generator(self.BatchGeneratorTrain.BatchGenerator(), val_samples=self.trainingDataset.size(), max_q_size=1, pickle_safe=False);
		return self.evaluationHistory;

	def validateModel(self):
		print("test model...");
		sum_predictions = 0;

		rand_samples = 10;
		for i in range(0,rand_samples):	
			predictions = self.model.predict_generator(self.BatchGeneratorVal.BatchGenerator(), val_samples=self.validationDataset.getNClasses(), max_q_size=1, pickle_safe=True);
			print("[" + str(i+1) + "]" + " -> " + str(predictions.shape));
			print(predictions[:5]);
			sum_predictions += predictions;
		test_predictions = sum_predictions / rand_samples;
		print(test_predictions.shape);

		# save predictions to csv
		self.saveValidationPredictionsToCSV(test_predictions);

	def testModel(self):
		print("test model...");
		sum_predictions = 0;

		rand_samples = 10;
		for i in range(0,rand_samples):	
			predictions = self.model.predict_generator(self.BatchGeneratorTest.BatchGenerator(), val_samples=self.testDataset.getNClasses(), max_q_size=1, pickle_safe=True);
			print("[" + str(i+1) + "]" + " -> " + str(predictions.shape));
			print(predictions[:5]);
			sum_predictions += predictions;
		test_predictions = sum_predictions / rand_samples;
		print(test_predictions.shape);

		# save predictions to csv
		self.saveTestPredictionsToCSV(test_predictions);
	
	def predict(self):
		predictions = self.model.predict_generator(self.BatchGeneratorVal.BatchGenerator(), val_samples=self.validationDataset.getNClasses(), max_q_size=1, pickle_safe=True);
		return predictions;
		
	def saveModel(self):
		# save model
		self.model.save(self.output_path + results + "model_final.h5");

	def loadModel(self, filename):
		# load model
		self.model = load_model(filename);	

	def saveModelArch(self):
		# save model architecture
		print("save model architecture to " + self.output_path + self.results);
		model_json = self.model.to_json();
		with open(self.output_path + self.results + "model.json", "w") as json_file:
		    json_file.write(model_json);

	def loadModelArch(self, filename):
		print("load model weights...");
		self.model = None;
		# load model
		json_file = open(filename, 'r');
		loaded_model_json = json_file.read();
		json_file.close();
		self.model = model_from_json(loaded_model_json, custom_objects={'init': self.initWeights});

	def loadWeights(self, filename):
		# load weights
		self.model.load_weights(filename);

	def saveTrainingHistoryToCSV(self):
		#save training history to csv file
		metricsFile = open(self.output_path + self.results + "metrics.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		#csvdata = zip(self.trainingHistory.history['loss'], self.trainingHistory.history['acc'], self.trainingHistory.history['val_loss'], self.trainingHistory.history['val_acc']);
		csvdata = zip(self.trainingHistory.history['loss'], self.trainingHistory.history['val_loss']);
		csvWriter.writerows(csvdata);
		metricsFile.close();	

	def savePredictionsToCSV(self):
		print("save predictions...");
		#save training history to csv file
		predictionsList = [];
		ground_truthList = [];
		vids = [];
		names = [];

		print("calculate predictions...");
		predictions = self.predict();
		print(predictions.shape);
		print(predictions[:30]);
		
		for i in range(0, self.validationDataset.getNClasses()):
			ids, n, f, l = self.validationDataset.getAllSamplesOfID(i+4801);
			names.append(n[0]);
			ground_truth_str = str(l[0][0]) + "," + str(l[0][1]) + "," + str(l[0][2]) + "," + str(l[0][3]) + "," + str(l[0][4]);
			ground_truthList.append(ground_truth_str);
			predictions_str = str(predictions[i][0]) + "," + str(predictions[i][1]) + "," + str(predictions[i][2]) + "," + str(predictions[i][3]) + "," + str(predictions[i][4]);  
			predictionsList.append(predictions_str);

		names = np.array(names);
		predictionsList = np.array(predictionsList);
		ground_truthList = np.array(ground_truthList);

		print("save video based predictions...");
		metricsFile = open(self.output_path + self.results + "predictions_video_based_val.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, predictionsList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

		metricsFile = open(self.output_path + self.results + "groundtruth_video_based_val.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, ground_truthList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

	def saveTrainingPredictionsToCSV(self):
		print("save predictions...");
		#save training history to csv file
		predictionsList = [];
		ground_truthList = [];
		vids = [];
		names = [];

		print("calculate predictions...");
		predictions = self.model.predict_generator(self.BatchGeneratorTrain.BatchGenerator(), val_samples=self.trainingDataset.size(), max_q_size=1, pickle_safe=False);

		for i in range(0, self.trainingDataset.size()):
			names.append(self.trainingDataset.getName(i));
			vids.append(self.trainingDataset.getVID(i));
			frame, ground_truth = self.trainingDataset.sample(i);	
			ground_truth_str = str(ground_truth[0]) + "," + str(ground_truth[1]) + "," + str(ground_truth[2]) + "," + str(ground_truth[3]) + "," + str(ground_truth[4]);
			ground_truthList.append(ground_truth_str);
			predictions_str = str(predictions[i][0]) + "," + str(predictions[i][1]) + "," + str(predictions[i][2]) + "," + str(predictions[i][3]) + "," + str(predictions[i][4]);  
			predictionsList.append(predictions_str);

		vids = np.array(vids);
		names = np.array(names);
		predictionsList = np.array(predictionsList);
		ground_truthList = np.array(ground_truthList);

		metricsFile = open(self.output_path + self.results + "predictions_image_based_train.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, predictionsList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

		metricsFile = open(self.output_path + self.results + "groundtruth_image_based_train.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, ground_truthList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

	def saveTestPredictionsToCSV(self, predictions):
		print("save predictions...");
		#save training history to csv file
		predictionsList = [];
		vids = [];
		names = [];

		print("calculate predictions...");
		#predictions = self.predict();
		print(predictions.shape);
		print(predictions[:30]);
		
		for i in range(0, self.testDataset.getNClasses()):
			ids, n, f = self.testDataset.getAllTestSamplesOfID(i+1);
			#print(ids.shape);
			if(ids.shape[0] <> 0):
				names.append(n[0]);
				predictions_str = str(predictions[i][0]) + "," + str(predictions[i][1]) + "," + str(predictions[i][2]) + "," + str(predictions[i][3]) + "," + str(predictions[i][4]);  
				predictionsList.append(predictions_str);

		names = np.array(names);
		predictionsList = np.array(predictionsList);

		print("save video based predictions...");
		metricsFile = open(self.output_path + self.results + "predictions_video_based_test.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, predictionsList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

	def saveValidationPredictionsToCSV(self, predictions):
		print("save predictions...");
		#save training history to csv file
		predictionsList = [];
		ground_truthList = [];
		vids = [];
		names = [];

		print("calculate predictions...");
		#predictions = self.predict();
		print(predictions.shape);
		print(predictions[:30]);

		for i in range(0, self.validationDataset.getNClasses()):
			ids, n, f, l = self.validationDataset.getAllSamplesOfID(i+4801);
			names.append(n[0]);
			ground_truth_str = str(l[0][0]) + "," + str(l[0][1]) + "," + str(l[0][2]) + "," + str(l[0][3]) + "," + str(l[0][4]);
			ground_truthList.append(ground_truth_str);
			predictions_str = str(predictions[i][0]) + "," + str(predictions[i][1]) + "," + str(predictions[i][2]) + "," + str(predictions[i][3]) + "," + str(predictions[i][4]);  
			predictionsList.append(predictions_str);

		names = np.array(names);
		predictionsList = np.array(predictionsList);
		ground_truthList = np.array(ground_truthList);

		print("save video based predictions...");
		metricsFile = open(self.output_path + self.results + "predictions_video_based_val.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, predictionsList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

		metricsFile = open(self.output_path + self.results + "groundtruth_video_based_val.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, ground_truthList);
		csvWriter.writerows(csvdata);
		metricsFile.close();
		
	def saveVideoPredictionsToCSV(self):
		print("calculate video based predictions...");
		
	

	def initWeights(self, shape, name=None, dim_ordering='tf'):
		return initializations.he_normal(shape, name=name, dim_ordering=dim_ordering);

	def selectOptimizer(self, optim):
		if(optim == 'sgd'):
			self.optim = SGD(lr=self.lRate, decay=self.lDecay, momentum=0.9, nesterov=True);
		elif(optim == 'adam'):		
			self.optim = Adam(lr=self.lRate, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, decay=self.lDecay);
		else:
			print("ERROR: Select correct optimizer!");
			exit();

	def setHyperparameters(self, params):
		
			if(len(params) == 4):
				self.lRate = params[0];
				self.lDecay = params[1];
				self.wDecay = params[2];
				self.pDropout = params[3];
			elif(len(params) == 7):
				self.lRate = params[0];
				self.lDecay = params[1];
				self.beta_1 = params[2];
				self.beta_2 = params[3];
				self.epsilon = params[4];
				self.wDecay = params[5];
				self.pDropout = params[6];
			else:
				print("ERROR: Select correct parameters!");
				exit();
		

	def printModelSummary(self):
		print(self.model.summary());
		print("\nused optimizer and hyperparameters: ");
		print("---------------------");
		print("Optimizer: " + str(self.optim));
		print("Learning rate: " + str(self.lRate));
		print("Learning rate decay: " + str(self.lDecay));
		print("epsilon: " + str(self.epsilon));
		print("beta 1: " + str(self.beta_1));
		print("beta 2: " + str(self.beta_2));
		print("Weight decay (L2): " + str(self.wDecay));
		print("Dropout p: " + str(self.pDropout));
		print("\n");

	def printTrainingHistory(self):
		print(self.trainingHistory);

	def getLayerOutput(self, X):
		f = [];
		get_layer_output = K.function([self.model.layers[0].input], [self.model.layers[5].output]);

		X ,y = self.validationDataset.sample(0);
		f.append(X);

		X = np.array(f);
		
		print(X.shape);

		layer_output = get_layer_output([X])[0];
		layer_output = np.squeeze(layer_output)
		print("layer_output shape : ", layer_output.shape)

		plt.imshow(layer_output[10]);
		plt.show();

	def createCallbacks(self):
		early_stopping = EarlyStopping(monitor='val_loss', patience=100);
		lrate = LearningRateScheduler(self.step_decay);
		bestModel = ModelCheckpoint(self.output_path + self.results + "weights_best.h5", 
						monitor='val_loss', 
						verbose=1, 
						save_best_only=True, 
						save_weights_only=True, 
						mode='auto');

		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, verbose=1, min_lr=1e-4)
		self.callbacks_list = [bestModel, reduce_lr, early_stopping]; #reduce_lr

	def step_decay(self, epoch):
		initial_lrate = 0.05;
		drop = 0.5;
		epochs_drop = 25.0;
		lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop));
		print("learning rate: " + str(lrate));
		return lrate;

	def selectRandomParams(self, optim):
		if(optim == 'sgd'):
			a = 0.1; #learning rate
			b = 0;   #round(random.uniform(0.00001, 0.000001), 7);   #learning decay
			c = round(random.uniform(0.01, 0.0005), 5);   #weight decay
			d = 0;   # round(random.uniform(0.0, 0.5), 1);   #dropout p
			print("[parameter] --> lr="+str(a)+", lr decay="+str(b)+", weight decay:"+str(c)+", dropout:"+str(d));
			params = [a, b, c, d];
			return params;
		elif(optim == 'adam'):		
			a = 0.1; #learning rate
			b = 0;   #learning decay
			c = 0;   #beta_1
			d = 0;   #beta_2
			e = 0;   #weight decay
			f = 0;   #dropout p
			print("[parameter] --> lr="+str(a)+", lr decay="+str(b)+ "beta_1="+str(c)+", beta_2="+str(d)+", weight decay:"+str(e)+", dropout:"+str(f));
			params = [a, b, c, d, e, f];
			return params;
		else:
			print("ERROR: Select correct optimizer!");
			exit();
	

	def getIntermediateLayerOutput(self, model, name, inputData):
		layer_name = name;
		intermediate_output = [];
		intermediate_layer_model = Model(input=model.input, output=model.get_layer(layer_name).output);
		intermediate_output = intermediate_layer_model.predict(inputData);
		return intermediate_output;


	def get_output_layer(self, model, layer_name):
		# get the symbolic outputs of each "key" layer (we gave them unique names).
		layer_dict = dict([(layer.name, layer) for layer in model.layers])
		layer = layer_dict[layer_name]
		return layer

	def visualize_class_activation_map(self):
		random_value = random.randint(0, 4000);
		original_img ,y = self.trainingDataset.getOrigSample(random_value);
		width, height, _ = original_img.shape
		x_batch = np.expand_dims(original_img,axis=0);

	def getLayerOutput(self):
		f = [];
		#idx = [19,137,51];
		#for i in range(0, 3):
	#		random_value = idx[i]; #random.randint(0, 200);
#			print("id: " + str(random_value));
#			X ,y = self.trainingDataset.getOrigSample(random_value);
#			
#			resized_image = cv2.resize(X, (X.shape[1]/2, X.shape[0]/2)) 
#			x_batch = np.expand_dims(resized_image,axis=0);
#			
#			pyplot.subplot(3, 6, (i*6+1))
#			pyplot.imshow(resized_image)
#			#pyplot.axis('off')
#
#			for a in range(1, 6):
#				pyplot.subplot(3, 6, (i*6+a+1))
#				layer_output = self.getIntermediateLayerOutput(self.model, "convolution2d_3", x_batch); #activation_1 convolution2d_1
#				output = np.squeeze(layer_output, axis=0);
#				offset = 12;
#				o = output[:,:,(a+offset):(a+1+offset)];
#				o = o.reshape(o.shape[:2]);
#				pyplot.imshow(o)
#				#pyplot.axis('off')
#
#		pyplot.show()

		random_value = 124; #random.randint(0, 200);
		print("id: " + str(random_value));

		r_idx = [];
		for a in range(0, 10):
			start = a*6;
			end = (a*6 + 6)-1;
			r_tmp = randint(start, end);
			r_idx.append(r_tmp);
		batch ,y = self.BatchGeneratorTrain.batch_Random(1, r_idx);
		X =batch[0];
		
		resized_image = X; #cv2.resize(X, (X.shape[1]/2, X.shape[0]/2)) 
		x_batch = np.expand_dims(resized_image,axis=0);
		print(x_batch.shape);	

		for a in range(1, 64):
			pyplot.subplot(8, 8, (1))
			pyplot.imshow(resized_image[0])
			pyplot.axis('off')
			pyplot.subplot(8, 8, (a+1))
			layer_output = self.getIntermediateLayerOutput(self.model, "activation_9", x_batch); #activation_1 convolution2d_1
			print(layer_output.shape)
			output = np.squeeze(layer_output, axis=0);
			offset = 0;
			o = output[:,:,:,(a+offset):(a+1+offset)];
			print(o.shape)
			o = o.reshape(o.shape[0:3]);
			pyplot.imshow(o[0])
			pyplot.axis('off')
		pyplot.show()

		#for i in range(0, 3):
		#	#X ,y = self.validationDataset.sample(50);
		#	random_value = random.randint(0, 4000);
		#	X ,y = self.trainingDataset.getOrigSample(random_value);
		#	resized_image = X#cv2.resize(X, (X.shape[1]/2, X.shape[0]/2)) 
		#	print(resized_image.shape);
		#	x_batch = np.expand_dims(resized_image,axis=0);
#
#			layer_output = self.getIntermediateLayerOutput(self.model, "maxpooling2d_4", x_batch); #activation_1 convolution2d_1
#			output = np.squeeze(layer_output, axis=0);
#			#print(output.shape);
#
#			pyplot.subplot(4, 3, (i+1))
#			o = output[:,:,i:(i+1)];
#			o = o.reshape(o.shape[:2]);
#			pyplot.imshow(resized_image)
#
#		#for i in range(0, 3):
#			pyplot.subplot(4, 3, (i+4))
#			layer_output = self.getIntermediateLayerOutput(self.model, "convolution2d_1", x_batch); #activation_1 convolution2d_1
#			output = np.squeeze(layer_output, axis=0);
#			o = output[:,:,:1];
#			o = o.reshape(o.shape[:2]);
#			pyplot.imshow(o)
#
#		#for i in range(0, 3):
#			pyplot.subplot(4, 3, (i+7))
#			layer_output = self.getIntermediateLayerOutput(self.model, "activation_1", x_batch); #activation_1 convolution2d_1
#			output = np.squeeze(layer_output, axis=0);
#			o = output[:,:,i:(i+1)];
#			o = o.reshape(o.shape[:2]);
#			pyplot.imshow(o)
#
#		#for i in range(0, 3):
#			pyplot.subplot(4, 3, (i+10))
#			layer_output = self.getIntermediateLayerOutput(self.model, "maxpooling2d_4", x_batch); #activation_1 convolution2d_1
#			output = np.squeeze(layer_output, axis=0);
#			o = output[:,:,i:(i+1)];
#			o = o.reshape(o.shape[:2]);
#			pyplot.imshow(o)
#		# show the plot
#		pyplot.show()


		

		#get_layer_output = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[5].output]);
		#layer_output = get_layer_output([f, 1])[0];
		#layer_output = np.squeeze(layer_output)
		#print("layer_output shape : " + str(layer_output.shape))
		#sumLayer = [];

		#for i in range(0,layer_output.shape[1]):
		#	plt.imshow(layer_output[0][i]);
		#	plt.show();	

	def generate_cam(self):
	
		#model = VGG16(weights='imagenet', include_top=True)
		print('Model loaded.')

		# The name of the layer we want to visualize
		# (see model definition in vggnet.py)
		layer_name = 'pool4'
		layer_idx = [idx for idx, layer in enumerate(self.model.layers) if layer.name == layer_name][0]

		for path in range(0,1):
			seed_img = utils.load_img("/media/administrator/Working/Daniel/Testvideos/images/dataAug_2.jpg", target_size=(64, 64))
			pred_class = np.argmax(self.model.predict(np.array([img_to_array(seed_img)])))
			heatmap = visualize_cam(self.model, layer_idx, [pred_class], seed_img)
			cv2.imshow('Attention - {}', heatmap)
			cv2.waitKey(0)

