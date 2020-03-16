from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
import numpy as np

sequences = 5;


vids_train = np.arange(48);
print(vids_train);

vids_val = np.arange(12);
print(vids_val);

def BatchGenerator(mode):
	if(mode == 'train'):
		while(1):
			for i in range(1, 49):
				vid = i;
				condition = (vids_train == vid);
				found_idx = np.where( condition );

				# get samples of VID
				train_x = np.random.random((10, 12, 12, 3));
				train_y = np.random.random((10, 5));

				
				
				s = np.zeros((sequences, 12, 12, 3));

				if(sequences > train_x.shape[0]):
					s[:train_x.shape[0],:,:,:] = train_x[:train_x.shape[0],:,:,:];
				elif(sequences <= train_x.shape[0]):
					s[:sequences,:,:,:] = train_x[:sequences,:,:,:];

				s = np.reshape(s, (1, sequences, 12, 12, 3));
				l = train_y[:1, :];
				
				#print("training");
				#print(vid);
				#print(train_x.shape);
				#print(train_y.shape);
				#print(s.shape);
				#print(l.shape);

				yield s, l;

	elif(mode == 'val'):
		while(1):
			for i in range(48, 61):
				vid = i;
				condition = (vids_val == vid);
				found_idx = np.where( condition );

				# get samples of VID
				val_x = np.random.random((10, 12, 12, 3));
				val_y = np.random.random((10, 5));
				
				s = np.zeros((sequences, 12, 12, 3));

				if(sequences > val_x.shape[0]):
					s[:val_x.shape[0],:,:,:] = val_x[:val_x.shape[0],:,:,:];
				elif(sequences <= val_x.shape[0]):
					s[:sequences,:,:,:] = val_x[:sequences,:,:,:];

				s = np.reshape(s, (1, sequences, 12, 12, 3));
				l = val_y[:1, :];
				
				#print("validation");
				#print(vid);
				#print(train_x.shape);
				#print(train_y.shape);
				#print(s.shape);
				#print(l.shape);
				
				yield s, l;


# load data

# create model
model = Sequential();
# 1st layer group
model.add(Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', name='conv1', input_shape=(sequences, 12, 12, 3)));
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)));
# 2nd layer group
model.add(Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', name='conv2'));
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)));
# 3rd layer group
#model.add(Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', name='conv3'));
#model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)));

model.add(Flatten())
# FC layers group
model.add(Dense(64, activation='relu', name='fc6'))
model.add(Dropout(0.0))
model.add(Dense(5, activation='sigmoid', name='fc8'))
print(model.summary())

model.compile(optimizer='sgd', loss='mse');


# train model
model.fit_generator(BatchGenerator('train'), 
                    samples_per_epoch=48, 
                    nb_epoch=5,
		    validation_data=BatchGenerator('val'), 
		    nb_val_samples=12,
		    nb_worker=1, 
  	            pickle_safe=False);




