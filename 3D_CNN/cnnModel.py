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


class CnnModel:
	output_path = "";
	BatchGeneratorTrain = None;
	BatchGeneratorVal = None;
	trainingDataset = None;
	validationDataset = None;
	testDataset = None;
	model = None;
	trainingHistory = None;

	nEpochs = 50;
	batchsize = 64;
	sequences = 6;

	sample_shape = [];

	callbacks_list = [];

	# hyperparameter
	lRate = 0;
	lDecay = 0;
	beta_1 = 0;
	beta_2 = 0;
	epsilon = 0;

	wDecay = 0;
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
		self.sample_shape[0] = self.sequences;
		self.sample_shape[1] = sample.shape[0];
		self.sample_shape[2] = sample.shape[1];
		self.sample_shape[3] = sample.shape[2];
		
		self.BatchGeneratorTrain = MiniBatchGenerator(self.trainingDataset, 'train', self.batchsize);
		self.BatchGeneratorVal = MiniBatchGenerator(self.validationDataset, 'val', self.batchsize);	
		self.BatchGeneratorTest = MiniBatchGenerator(self.testDataset, 'test', self.batchsize);	
		
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
		self.model.add(Convolution3D(64, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv1a', input_shape=(self.sample_shape[0], self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));

		self.model.add(Convolution3D(64, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv1b'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool1'));

		# 2nd layer group
		self.model.add(Convolution3D(128, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv2a'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));

		self.model.add(Convolution3D(128, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv2b'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool2'));

		# 3rd layer group
		self.model.add(Convolution3D(256, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv3a'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));

		self.model.add(Convolution3D(256, 3, 3, 3, W_regularizer=l2(self.wDecay), init=self.initWeights, border_mode='same', name='conv3b'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		self.model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool3'));
		
		self.model.add(Flatten());
		
		# FC layers group
		self.model.add(Dense(1024, W_regularizer=l2(self.wDecay), name='fc1'));
		self.model.add(BatchNormalization(mode=batchMode));
		self.model.add(Activation('relu'));
		self.model.add(Dropout(self.pDropout));

		self.model.add(Dense(5, activation='sigmoid', name='fc2'));

	def compileModel(self):
		print("compile model...");
		self.model.compile(optimizer=self.optim, loss='mse');	#, metrics=['accuracy']

	def smooth_huber_loss(self, y_true, y_pred):
		    #"""Regression loss function, smooth version of Huber loss function. """
		W = 1.5;
		
		cosh = (K.exp(y_true - y_pred) + K.exp((-1) * (y_true - y_pred))) / 2;

		return K.mean(w * K.log(cosh));

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
							samples_per_epoch=self.trainingDataset.size(), 
							nb_epoch=self.nEpochs,  
							max_q_size=1, 
							pickle_safe=False, 
							verbose=1, 
							callbacks=self.callbacks_list, 
							validation_data=self.BatchGeneratorVal.BatchGenerator(), 
							nb_val_samples=self.validationDataset.size());	
		return self.trainingHistory;		


	def evaluateModel(self):
		print("evaluate model...");
		self.evaluationHistory = self.model.evaluate_generator(self.BatchGeneratorVal.BatchGenerator(), val_samples=self.validationDataset.size(), max_q_size=1, pickle_safe=False);
		#self.evaluationHistory = self.model.evaluate_generator(self.BatchGeneratorTrain.BatchGenerator(), val_samples=self.trainingDataset.size(), max_q_size=1, pickle_safe=False);
		return self.evaluationHistory;

	def testModel(self):
		print("test model...");
		#test_predictions = self.model.predict_generator(self.BatchGeneratorTest.BatchGenerator(), val_samples=self.testDataset.size(), max_q_size=1, pickle_safe=False);
		test_predictions = [];
		#print(self.testDataset.size());

		for i in range(0, self.testDataset.size()):
			samples = self.testDataset.sample(i);
			preds = self.model.predict_on_batch(samples);
			preds_avg = np.mean(preds, axis=0);
			test_predictions.append(preds_avg);
			#print(samples.shape);
			#print(preds);
			#print(preds_avg);

		# save predictions to csv
		self.saveTestPredictionsToCSV(test_predictions);
	
	def predict(self):
		predictions = self.model.predict_generator(self.BatchGeneratorVal.BatchGenerator(), val_samples=self.validationDataset.size(), max_q_size=1, pickle_safe=False);
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

		for i in range(0, self.validationDataset.size()):
			names.append(self.validationDataset.getName(i));
			vids.append(self.validationDataset.getVID(i));
			frame, ground_truth = self.validationDataset.sample(i);	
			ground_truth_str = str(ground_truth[0]) + "," + str(ground_truth[1]) + "," + str(ground_truth[2]) + "," + str(ground_truth[3]) + "," + str(ground_truth[4]);
			ground_truthList.append(ground_truth_str);
			predictions_str = str(predictions[i][0]) + "," + str(predictions[i][1]) + "," + str(predictions[i][2]) + "," + str(predictions[i][3]) + "," + str(predictions[i][4]);  
			predictionsList.append(predictions_str);

		vids = np.array(vids);
		names = np.array(names);
		predictionsList = np.array(predictionsList);
		ground_truthList = np.array(ground_truthList);

		metricsFile = open(self.output_path + self.results + "predictions_image_based_val.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, predictionsList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

		metricsFile = open(self.output_path + self.results + "groundtruth_image_based_val.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, ground_truthList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

		print("calculate video based predictions...");
		predictionsList_n = [];
		ground_truthList_n = [];
		vidsList_n = [];
		namesList_n = [];

		for i in range(np.min(vids), np.max(vids)+1):
			#print("id: " + str(i));
			#print("vids: " + str(vids));
			condition = (vids==i);
			foundIdx = np.where(condition);
			#print("idx: " + str(foundIdx));
			vids_n = vids[foundIdx];
			names_n = names[foundIdx];
			ground_truth_n = ground_truthList[foundIdx];
			
			predictions_n = predictionsList[foundIdx];

			pred_n = [];
			ground_n = [];
			pred_n_f = [];
			for a in range(0, predictions_n.shape[0]):
				s = np.fromstring(predictions_n[a], dtype='float32', sep=',');
				pred_n_f.append(s);
				v = np.fromstring(ground_truth_n[a], dtype='float32', sep=',');
				ground_truth_str = str(v[0]) + "," + str(v[1]) + "," + str(v[2]) + "," + str(v[3]) + "," + str(v[4]);
				ground_n.append(ground_truth_str);

			ground_n = np.array(ground_n);
			predictions_avg = np.mean(pred_n_f, axis=0);
			predictionsAvg_str = str(predictions_avg[0]) + "," + str(predictions_avg[1]) + "," + str(predictions_avg[2]) + "," + str(predictions_avg[3]) + "," + str(predictions_avg[4]);  
			
			vidsList_n.append(vids_n[0]);
			namesList_n.append(names_n[0]);
			predictionsList_n.append(predictionsAvg_str);
			ground_truthList_n.append(ground_n[0]);
			
		vidsList_n = np.array(vidsList_n);
		namesList_n = np.array(namesList_n);
		ground_truthList_n = np.array(ground_truthList_n);
		predictionsList_n = np.array(predictionsList_n);

		print("save video based predictions...");
		metricsFile = open(self.output_path + self.results + "predictions_video_based_val.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(namesList_n, predictionsList_n);
		csvWriter.writerows(csvdata);
		metricsFile.close();

		metricsFile = open(self.output_path + self.results + "groundtruth_video_based_val.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(namesList_n, ground_truthList_n);
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
		#print(predictions);
		#predictions = self.predict();

		for i in range(0, self.testDataset.size()):
			names.append(self.testDataset.getName(i));
			vids.append(self.testDataset.getVID(i));
			frame = self.testDataset.sample(i);	
			predictions_str = str(predictions[i][0]) + "," + str(predictions[i][1]) + "," + str(predictions[i][2]) + "," + str(predictions[i][3]) + "," + str(predictions[i][4]);  
			predictionsList.append(predictions_str);

		vids = np.array(vids);
		names = np.array(names);
		predictionsList = np.array(predictionsList);
		
		metricsFile = open(self.output_path + self.results + "predictions_image_based_test.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(names, predictionsList);
		csvWriter.writerows(csvdata);
		metricsFile.close();

		print("calculate video based predictions...");
		predictionsList_n = [];
		vidsList_n = [];
		namesList_n = [];
		#print(str(np.min(vids)) + "," + str(np.max(vids)));
		for i in range(np.min(vids), np.max(vids)+1):
			print("id: " + str(i));
			#print("vids: " + str(vids));
			condition = (vids==i);
			foundIdx = np.where(condition);
			print(foundIdx);
			tmp = np.any(condition);
			if(tmp <> False ):
				#print("idx: " + str(foundIdx));
				vids_n = vids[foundIdx];
				names_n = names[foundIdx];
			
				predictions_n = predictionsList[foundIdx];

				pred_n = [];
				pred_n_f = [];
				for a in range(0, predictions_n.shape[0]):
					s = np.fromstring(predictions_n[a], dtype='float32', sep=',');
					pred_n_f.append(s);

				predictions_avg = np.mean(pred_n_f, axis=0);
				predictionsAvg_str = str(predictions_avg[0]) + "," + str(predictions_avg[1]) + "," + str(predictions_avg[2]) + "," + str(predictions_avg[3]) + "," + str(predictions_avg[4]);  
			
				vidsList_n.append(vids_n[0]);
				namesList_n.append(names_n[0]);
				predictionsList_n.append(predictionsAvg_str);
			
		vidsList_n = np.array(vidsList_n);
		namesList_n = np.array(namesList_n);
		predictionsList_n = np.array(predictionsList_n);

		print("save video based test_predictions...");
		metricsFile = open(self.output_path + self.results + "predictions_video_based_test.csv", 'wb');
		csvWriter = csv.writer(metricsFile);
		csvdata = zip(namesList_n, predictionsList_n);
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

		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, min_lr=1e-5)
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
		

