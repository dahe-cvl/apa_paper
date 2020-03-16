import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


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

		self.datagen.fit(batch[0]);
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


	def showImages(self, batch):
		for i in range(0, 9):
			pyplot.subplot(330 + 1 + i)
			pyplot.imshow(batch[0][i])
		# show the plot
		pyplot.show()
		self.datagen.fit(batch[0]);
		for X_batch, y_batch in self.datagen.flow(batch[0], batch[1], batch_size=64):
			#print(X_batch.shape);
			#print(y_batch.shape);
			# create a grid of 3x3 images
			for i in range(0, 9):
				pyplot.subplot(330 + 1 + i)
				pyplot.imshow(X_batch[i])
			# show the plot
			pyplot.show()
			break
