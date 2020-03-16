import os
import sys
import getopt
import csv
import numpy as np
import zipfile, glob
import cv2,shutil
import h5py as hf
import matplotlib.pyplot as plt

from faceDetector import FaceDetection
from imageExtractor import ImageExtractor

tmp_path = "tmp/";

fpsNew = 2;
cols = 64;
rows = 64;

database = 0;
size_v_old = 0;
size_n_old = 0;
size_f_old = 0;
size_l_old = 0;

def runOnVideo():
	input_path = "/media/administrator/Working/Daniel/Masterarbeit/Implementierung/Masterarbeit/FaceDBGenerator_V2/Facedetector/";
	filename = "CFK8ib0aWe8.000.mp4";
	print("create image extractor...");
	imageExtractor = ImageExtractor(input_path);

	file = open(filename, 'r');
	print("-------------------------");
	print(filename);

	ret = imageExtractor.loadVideo(input_path + filename);
	if(ret == True):			
		imageExtractor.reduceFrameRate(filename, 4);

		frames = imageExtractor.getFrames();
		labels = imageExtractor.getLabels();
		videonames = imageExtractor.getVideonames();
		vids = imageExtractor.getVids();
	print("-------------------------");


def run():
	print("run...");

	print("create image extractor...");
	imageExtractor = ImageExtractor(input_path);

	# init hdf5 database
	dataset_size = 100 * 75 * 80;
	hdf5file = output_path + "database_TEST";
	initHDF5Database(hdf5file, dataset_size);	

	print("run image extractor...");

	# extract videofiles to tmp folder
	cnt = 0;
	for i in range(15, 16, 1):
		cnt = cnt + 1;
		
		# load training data
		print("process training80_" + str(i).zfill(2)  + ".zip ...");
		filename = "training80_" + str(i).zfill(2)  + ".zip";

		try:
			zipArchive = zipfile.ZipFile(input_path + filename, "r");	
		except:
			print("ERROR: file not found!");
			ret = False;

		if not os.path.exists(tmp_path):
	    		os.mkdir( "tmp", 0755 );

		zipArchive.extractall(tmp_path);
		zipArchive.close();
	

		# videofiles in tmp folder
		frames = [];
		labels = [];
		videonames = [];
		vids = [];
		#filenames = glob.glob(tmp_path + "/*.mp4");
		filenames = os.listdir(tmp_path);
		#print(filenames);
		for filename in filenames:
			if filename.endswith(".mp4"):
        			#print(file)

				file = open(tmp_path + filename, 'r');
				print("-------------------------");
				print(filename);
			
				ret = imageExtractor.loadVideo(tmp_path + filename);
				if(ret == True):			
					imageExtractor.reduceFrameRate(filename, fpsNew);

					frames = imageExtractor.getFrames();
					labels = imageExtractor.getLabels();
					videonames = imageExtractor.getVideonames();
					vids = imageExtractor.getVids();
				print("-------------------------");
				
			#imageExtractor.printVideoDetails();
			#imageExtractor.playExtractedFrames();

		# delete tmp folder
		files = glob.glob(tmp_path + "/*");
		for name in files:
			os.remove(name);
		files = glob.glob(tmp_path + "/.*");
		for name in files:
			os.remove(name);
		#shutil.rmtree("/tmp", ignore_errors=False, onerror=None);
		print("number of frames: " + str(len(frames)));
		print("number of labels: " + str(len(labels)));
		print("number of vids: " + str(len(vids)));
		print("number of videonames: " + str(len(videonames)));
	
		#save to numpy array
		saveDataAsHDF5(output_path + "database_part1", videonames, vids, frames, labels);
		imageExtractor.frames = [];
		imageExtractor.labels = [];
		imageExtractor.vids = [];
		imageExtractor.videonames = [];
		frames = [];
		labels = [];
		vids = [];
		videonames = [];

def initHDF5Database(filename, dataset_size):
	global database;

	database = hf.File(filename + ".h5", 'a');
	database.create_dataset('id', (dataset_size,), dtype='int32');
	database.create_dataset('name', (dataset_size,), dtype='S30');
	database.create_dataset('data', (dataset_size,rows,cols,3), dtype='uint8');
	database.create_dataset('labels', (dataset_size,5), dtype='float32');

	database.close();


def saveDataAsHDF5(filename, n, v, f, l):
	global size_v_old;
	global size_n_old;
	global size_f_old;
	global size_l_old;

	print("save to hdf5");
	database = hf.File(filename + ".h5", 'a');

	v_numpy = np.array(v);
	v_numpy = v_numpy.astype('int32');

	n_numpy = np.array(n);
	n_numpy = n_numpy.astype('string');

	f_numpy = np.array(f);
	f_numpy = f_numpy.astype('uint8');

	l_numpy = np.array(l);
	l_numpy = l_numpy.astype('float32');

	size_v = size_v_old + v_numpy.shape[0];
	size_n = size_n_old + n_numpy.shape[0];
	size_f = size_f_old + f_numpy.shape[0];
	size_l = size_l_old + l_numpy.shape[0];

	print("shuffle");
	shuffled_name, shuffled_vids, shuffled_f, shuffled_l = shuffleDataset(n_numpy, v_numpy, f_numpy, l_numpy);

	print("save to database");
	database["id"][int(size_v_old):int(size_v)] = shuffled_vids;
	database["name"][int(size_n_old):int(size_n)] = shuffled_name;
	database["data"][int(size_f_old):int(size_f)] = shuffled_f;	
	database["labels"][int(size_f_old):int(size_f)] = shuffled_l;

	size_v_old = size_v;
	size_n_old = size_n;
	size_f_old = size_f;
	size_l_old = size_l;

	database.close();

def shuffleDataset(name, vid, f, l):
	datasize = f.shape[0];
	print(datasize);	
	randomize = np.arange(datasize);
	np.random.shuffle(randomize);
	shuffled_name = name[randomize];
	shuffled_vid = vid[randomize];
	shuffled_f = f[randomize];
	shuffled_l = l[randomize];
	return shuffled_name, shuffled_vid, shuffled_f, shuffled_l;

def getCmdParameter(argv):
	ipath = "";
   	opath = "";

	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="]);
	except getopt.GetoptError:
		print("ERROR: false input!");
		print("Example: preProcessing_v1.py -i <inputfile> -o <outputfile> ");
		sys.exit();

	for opt, arg in opts:
		if opt == '-h':
			print("Example: preProcessing_v1.py -i <inputfile> -o <outputfile> ");
			sys.exit();
		elif opt in ("-i", "--ifile"):
			ipath = arg;
		elif opt in ("-o", "--ofile"):
			opath = arg;

	print ("Input path: " + str(ipath));
	print ("Output path: " + str(opath));	
	
	return ipath, opath;

def main(argv):
	global input_path;
	global output_path;

	# read input parameters
	input_path, output_path = getCmdParameter(argv);

	if (input_path == "" or output_path == ""):
		print("ERROR: false input!");
		print("Example: preProcessing_v2.py -i <inputfile> -o <outputfile>");
		sys.exit();
	else:
		#run();
		runOnVideo();	
		
	
if __name__ == "__main__":
    main(sys.argv[1:]);
