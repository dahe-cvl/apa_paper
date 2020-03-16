import sys, csv, math, random, os
import numpy as np
from Cnn3DModel import Cnn3DModel
from FaceDataset import FaceDataset
from ExtendedFeatureDataset import ExtendedFeatureDataset
from FaceDataset_tiny import FaceDataset_tiny
from FaceDataset_tiny_classifier import FaceDataset_tiny_classifier
from multiprocessing import Process, Queue
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.models import model_from_json

from multiprocessing import Process, Queue

class runModel:
	trainingDataset = None;
	validationDataset = None;
	testDataset = None;

	output_path = "";
	input_path = "";
	model = None;

	def __init__(self, input_path, output_path):
		print("create instance...");
		self.output_path = output_path;
		self.input_path = input_path;

		# load dataset
		print("load dataset...");
		#self.trainingDataset = FaceDataset_tiny(self.input_path, "train_database_complete_shuffled.h5", "train");
		#self.validationDataset = FaceDataset_tiny(self.input_path, "val_database_complete_shuffled.h5", "val");
		#self.testDataset = FaceDataset_tiny(self.input_path, "test_database_complete.h5", "test");
				
		self.trainingDataset = FaceDataset(self.input_path, "train_database_complete.h5", "train");
		self.validationDataset = FaceDataset(self.input_path, "val_database_complete.h5", "val");
		self.testDataset = FaceDataset(self.input_path, "test_database_complete.h5", "test");

		#self.trainingDataset = ExtendedFeatureDataset(self.input_path, "train_database_complete_shuffled.h5", "train");
		#self.validationDataset = ExtendedFeatureDataset(self.input_path, "val_database_complete_shuffled.h5", "val");
		#self.testDataset = ExtendedFeatureDataset(self.input_path, "test_database_complete.h5", "test");
	
		trainingDatasetSamples = self.trainingDataset.size();
		print("trainingset sample size: " + str(trainingDatasetSamples));

		validationDatasetSamples = self.validationDataset.size();
		print("validationset sample size: " + str(validationDatasetSamples));

		testDatasetSamples = self.testDataset.size();
		print("testset sample size: " + str(testDatasetSamples));

		# create CNNModel instance
		self.model = Cnn3DModel(self.trainingDataset, self.validationDataset, self.testDataset, self.output_path);	

	def runTrainingMode(self):
		#set hyperparameters
		params = [0.1, 0, 0.002, 0.2];
		#params = [0.1, 0, 0.0, 0.0];
		#params = [0.1, 0, 0.9, 0.999, 1e-8, 0.002, 0.0];
		self.model.setHyperparameters(params);
		
		#select optimizer
		self.model.selectOptimizer("sgd");
		#self.model.selectOptimizer("adam");

		# create model
		self.model.createModel();

		#save model architecture to file
		self.model.saveModelArch();

		# compile model
		self.model.compileModel();
		
		# print model summary
		self.model.printModelSummary();

		# create callback functions
		self.model.createCallbacks();

		# train model
		history = self.model.trainModel();

		# save history to csv
		self.model.saveTrainingHistoryToCSV();

		#evaluate final model
		#scores = self.model.evaluateModel();
		#print(scores);

		# save predictions to csv
		self.model.savePredictionsToCSV();

	def runAdditionalTrainingMode(self, path):

		# load architecture
		self.model.loadModelArch(path + "Model/model.json");

		#load model weights
		self.model.loadWeights(path + "Model/weights_best.h5");

		#set hyperparameters
		params = [0.001, 0, 0.002, 0.2];
		self.model.setHyperparameters(params);
		
		#select optimizer
		self.model.selectOptimizer("sgd");

		# compile model
		self.model.compileModel();
		
		# print model summary
		self.model.printModelSummary();

		# create callback functions
		self.model.createCallbacks();

		# train model
		history = self.model.trainModel();

		# save history to csv
		self.model.saveTrainingHistoryToCSV();

		#evaluate final model
		scores = self.model.evaluateModel();
		print(scores);

		# save predictions to csv
		self.model.savePredictionsToCSV();


	def runTestMode(self):
		#print("NOT IMPLEMENTED");
		print("load model architecture...");
		self.model.loadModelArch(self.output_path + "Model/model.json");

		print("load best model weights...");
		self.model.loadWeights(self.output_path + "Model/weights_best.h5");

		# compile model
		self.model.compileModel();

		#print("test model...");
		#self.model.testModel();
		self.model.validateModel();

	def calculateTrainingsPredictions(self):
		print("load model architecture...");
		self.model.loadModelArch(self.output_path + "Models/model.json");

		print("load best model weights...");
		self.model.loadWeights(self.output_path + "Models/weights_best.h5");

		# compile model
		self.model.compileModel();

		# calcuate predictions
		self.model.saveTrainingPredictionsToCSV();

	def runEvaluateBestModel(self):
		print("find best model");
		scores = [];
		directories = os.listdir(self.output_path);
		print(directories);
		for directory in directories:
			print(directory);
			print("load model architecture from " + str(directory));

			# load architecture
			self.model.loadModelArch(self.output_path + directory + "/model.json");

			#load model weights
			self.model.loadWeights(self.output_path + directory + "/weights_best.h5");

			# compile model
			self.model.compileModel();

			# evaluate model
			scores.append(self.model.evaluateModel());

			# save predictions to csv
			self.model.savePredictionsToCSV();
		
		scores = np.array(scores);
		idx = scores[:,1].argmax();
		scores_best = scores[idx] * 100.0;
		print("directory name: " + str(directories[idx]));
		print("---------------------------------------");
		print("evaluation accuracy of best model: " + str(scores_best[1]) + "%");
		print("evaluation error of best model: " + str(scores_best[0]) + "%");

		
	def runHyperparameterSearch(self, nCombinations):
		for i in range(0, nCombinations):
			# select random params
			params = self.model.selectRandomParams('sgd');		
			self.run_in_separate_process(self.trainModelCombinations, (params, ));
			
	def trainModelCombinations(self, params):
			#set hyperparameters
			self.model.setHyperparameters(params);

			#select optimizer
			self.model.selectOptimizer('sgd');
		
			# create model
			self.model.createModel();

			#save model architecture to file
			self.model.saveModelArch();

			# compile model
			self.model.compileModel();

			# create callback functions
			self.model.createCallbacks();

			# train model
			history = self.model.trainModel();

			# save history to csv
			self.model.saveTrainingHistoryToCSV();

			# save predictions to csv
			self.model.savePredictionsToCSV();


	def run_in_separate_process(self, method, args):
		def queue_wrapper(q, params):
			r = method(*params)
			q.put(r)

		q = Queue()
		p = Process(target=queue_wrapper, args=(q, args))
		p.start()
		#return_val = q.get()
		p.join()
		
	def visTest(self):
		print("load model architecture...");
		self.model.loadModelArch(self.output_path + "Model/model.json");

		print("load best model weights...");
		self.model.loadWeights(self.output_path + "Model/weights_best.h5");

		# compile model
		self.model.compileModel();

		
		self.model.generate_cam();


	def run_visualization(self):
		# load architecture
		print("load model architecture...");
		#self.model.loadModelArch(self.output_path + "Model/model.json");
		
		self.model.addDecoder();
		
		# compile model
		self.model.compileModel();

		# print model summary
		self.model.printModelSummary();

		print("visualize layer outputs");
		self.model.getLayerOutput();
		#self.model.visualize_class_activation_map();


