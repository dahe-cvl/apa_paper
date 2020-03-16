import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2

class DataAugmentation:
	shift = 0.15;
	datagen = None;

	# constructor
	def __init__(self):
		# define data preparation
		self.datagen = ImageDataGenerator(featurewise_center=False,
						    samplewise_center=False,
						    featurewise_std_normalization=False,
						    samplewise_std_normalization=False,
						    zca_whitening=False,
						    rotation_range=0.,
						    width_shift_range=self.shift,
						    height_shift_range=self.shift,
						    shear_range=0.,
						    zoom_range=0.,
						    channel_shift_range=0.,
						    fill_mode='nearest',
						    cval=0.,
						    horizontal_flip=True,
						    vertical_flip=False,
						    rescale=None,
						    dim_ordering="tf");

	def runDataAugmentation(self, batch):
		frames = [];
		labels = [];

		#self.datagen.fit(batch[0]);
		for f, l in self.datagen.flow(batch[0], batch[1], batch_size=64):
			frames.append(f);
			labels.append(l);
			break;
		frames = np.array(frames);
		labels = np.array(labels);
		frames = np.squeeze(frames);		
		labels = np.squeeze(labels);
		#print(frames.shape);
		#print(labels.shape);
		return frames, labels;


	def runOverSampling(self, sample):
		s = [];

		# crops
		s1 = sample;	
		s2 = sample[0:56, 0:56];
		s3 = sample[(64-56):64, 0:56];
		s4 = sample[(64-56):64, (64-56):64];
		s5 = sample[0:56, (64-56):64];

		# mirrors
		s6 = cv2.flip(sample, 0);
		s7 = cv2.flip(sample, 1);
		s8 = cv2.flip(s6, 1);
		s9 = cv2.flip(s7, 1);

		# center
		s10 = sample[4:60, 4:60];

		s.append(s1);
		s.append(s2);
		s.append(s3);
		s.append(s4);
		s.append(s5);
		s.append(s6);
		s.append(s7);
		s.append(s8);
		s.append(s9);
		s.append(s10);

		s_resized = [];
		for i in range(0, len(s)):
			s_resized.append(cv2.resize(s[i], (64, 64)));

		tmp = np.array(s_resized);
		return tmp;

	def showImages(self, batch):
		#for i in range(0, 64):
		#	pyplot.subplot(8, 8, i+1)
		#	pyplot.imshow(batch[0][i])
		# show the plot
		#pyplot.show()
		#self.datagen.fit(batch[0]);
		for X_batch, y_batch in self.datagen.flow(batch[0], batch[1], batch_size=6):
			#print(X_batch.shape);
			#print(y_batch.shape);
			# create a grid of 3x3 images
			for i in range(0, 6):
				pyplot.subplot(2, 3, (i+1))
				pyplot.imshow(X_batch[i])
			# show the plot
			pyplot.show()
			break
