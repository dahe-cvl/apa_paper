from keras.utils.visualize_util import plot
from matplotlib import pyplot
import numpy as np
import cv2

class Visualization:
	# members	

	def __init__(self, output_path):
		self.output_path = output_path;
		
	def visualizeModelArchitecture(self, model):	
		print("save model...");	
		plot(model, to_file=self.output_path + "modelVisualization.png", show_shapes=True);

	def plotOriginalImageset(self, dataset):
		print("plot 9 images...");
		# plot 3 images
		
		#print(samples.shape);
		#print(samples[0, 0]);

		# create a grid of 3x3 images
		for i in range(0, 9):
			sample = (dataset.sample(i));
			#sample = np.array(sample);
			pyplot.subplot(330 + 1 + i);
			pyplot.imshow(cv2.cvtColor(sample[0], cv2.COLOR_BGR2RGB));
		# show the plot
		pyplot.show();

	def plotZeroCenteredImageset(self, dataset):
		print("plot 9 images...");
		# plot 3 images
		
		#print(samples.shape);
		#print(samples[0, 0]);

		# create a grid of 3x3 images
		for i in range(0, 9):
			sample = (dataset.sample(i));
			#sample = np.array(sample);
			pyplot.subplot(330 + 1 + i);
			pyplot.imshow(sample[0]);
		# show the plot
		pyplot.show();

	def plotNormalizedImageset(self, dataset):
		print("plot 9 images...");
		# plot 3 images
		
		#print(samples.shape);
		#print(samples[0, 0]);

		# create a grid of 3x3 images
		for i in range(0, 9):
			sample = (dataset.sample(i));
			maxValue = np.max(np.array(sample[0]));
			minValue = np.min(np.array(sample[0]));
			print(maxValue);
			print(minValue);
			#sample = np.array(sample);
			pyplot.subplot(330 + 1 + i);
			pyplot.imshow(sample[0]);
		# show the plot
		pyplot.show();
		
		
	def plotImageset(self, dataset1, dataset2, dataset3):
		for i in range(0, 3):
			sample = (dataset1.sample(i));
			#sample = np.array(sample);
			pyplot.subplot(330 + 1 + i);
			pyplot.imshow(cv2.cvtColor(sample[0], cv2.COLOR_BGR2RGB));

		for i in range(0, 3):
			sample = (dataset2.sample(i));
			#sample = np.array(sample);
			pyplot.subplot(330 + 1 + (3+i));
			pyplot.imshow(sample[0]);

		for i in range(0, 3):
			sample = (dataset3.sample(i));
			#sample = np.array(sample);
			pyplot.subplot(330 + 1 + (6+i));
			pyplot.imshow(sample[0]);
		# show the plot
		pyplot.show();	
		
	
