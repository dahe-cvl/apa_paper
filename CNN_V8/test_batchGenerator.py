from MiniBatchGenerator import MiniBatchGenerator
import sys, csv, math, random, os
import numpy as np
from FaceDataset import FaceDataset

#output_path = "/media/administrator/Working/Daniel/PreprocessedInput/FaceDatabase/80_images_per_video/";
input_path = "/media/administrator/Working/Daniel/PreprocessedInput/FaceDatabase/20_images_per_video/";

# load dataset
print("load dataset...");
trainingDataset = FaceDataset(input_path, "database_normalized_shuffled.h5", "train");
validationDataset = FaceDataset(input_path, "database_normalized_shuffled.h5", "val");

trainingDatasetSamples = trainingDataset.size();
print("trainingset sample size: " + str(trainingDatasetSamples));

validationDatasetSamples = validationDataset.size();
print("validationset sample size: " + str(validationDatasetSamples));

BatchGeneratorTrain = MiniBatchGenerator(trainingDataset, 32);
BatchGeneratorVal = MiniBatchGenerator(validationDataset, 32);

print(BatchGeneratorTrain.nbatches());

b = BatchGeneratorTrain.BatchGenerator();

