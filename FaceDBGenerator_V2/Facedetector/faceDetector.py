import sys
import os,math
import dlib
import glob
from skimage import io
import cv2
import numpy as np

class FaceDetection:
	#members
	predictor_path = "";
	detector = 0;
	predictor = 0;
	
	shape = 0;
	dets = 0;
	rects = 0;

	box_l = 0;
	box_r = 0;
	box_top = 0;
	box_bottom = 0;

	rows = 64;
	cols = 64;
		
	def __init__(self, predictor_path):
		print("create instance...");
		self.initFaceDetector(predictor_path);

	
	def initFaceDetector(self, predictor_path):
		self.detector = dlib.get_frontal_face_detector();
		self.predictor = dlib.shape_predictor(predictor_path);

	def rotateFace(self, image, center, rotation):
		mat = cv2.getRotationMatrix2D(tuple(np.array(np.array(center))), rotation, 1.0);
		#print(mat);
		#print(image.shape);

		h = image.shape[0];
		w = image.shape[1];
		sz = (w,h);
	    	
		#print("size: " + str(sz));
		image_rot = cv2.warpAffine(image, mat, sz);
		#print(image_rot.shape);

		return image_rot;

	def createBoundingBox(self, landmarks, image, distance):
		margin = 0;

		# create bounding box
		p27_x = landmarks[27,0];
		p27_y = landmarks[27,1];
		#print("landmark 27: " + str(p27_x) + ", " + str(p27_y));
	
		p8_x = landmarks[8,0];
		p8_y = landmarks[8,1] + margin;
		p15_x = landmarks[15,0]  + margin;
		p15_y = landmarks[15,1];
		p1_x = landmarks[1,0] - margin;
		p1_y = landmarks[1,1];
		P_x = p27_x;
		P_y = p27_y - distance - margin;

		if(P_y < 0):
			P_y = 0;
		if(p8_y > image.shape[0]):
			p8_y = image.shape[0];
		if(p1_x < 0):
			p1_x = 0;
		if(p15_x > image.shape[1]):
			p15_x = image.shape[1];
	
		#print("point top: " + str(P_x) + ", " + str(P_y));
		#print("point bottom: " + str(p8_x) + ", " + str(p8_y));
		#print("point right: " + str(p15_x) + ", " + str(p15_y));
		#print("point left: " + str(p1_x) + ", " + str(p1_y));
		
		p_t = P_y;
		p_b = p8_y;
		p_l = p1_x;
		p_r = p15_x;

		return p_t, p_b, p_l, p_r;


	def run(self, image):
		# get landmarks
		landmarks = self.get_landmarks(image);

		# get eye points
		eye_left, eye_right = self.calculateEyePoints(landmarks);

		# rotate image
		rotation = (-1) * self.estimateRollAngle(eye_left, eye_right);
		image_rot = self.rotateFace(image, eye_left, rotation);

		# calculate distance between the centers of the eyes
		distance = self.calculateDistance(eye_left, eye_right);
		#print("distance: " + str(distance));

		# get new landmarks
		landmarksNew = self.get_landmarks(image_rot);

		# create bounding box
		p_t, p_b, p_l, p_r = self.createBoundingBox(landmarksNew, image, distance);

		#crop image
		image_crop = image_rot[int(p_t):int(p_b), int(p_l):int(p_r)];	

		# resize image
		#print(	image_crop.shape);
		image_resized = cv2.resize(image_crop, (self.rows, self.cols)); 
		#print(image_resized.shape);

		#display images
		#self.showDetectedFace(image);
		#self.showDetectedFace(image_rot);
		#self.showDetectedFace(image_crop);
		#self.showDetectedFace(image_resized);

		return image_resized;

	def calculateDistance(self, p1, p2):
		dx = p2[0] - p1[0];
		dy = p2[1] - p1[1];
		return math.sqrt(dx*dx + dy*dy);

	def calculateEyePoints(self, landmarks):
		#Find the center of the left eye by averaging the points around the eye.
		cnt = 0;
		l_x  = 0;
		l_y = 0;
		for i in range(36, 41, 1):
			#print(str(landmarks[i,0]) + ", "+ str(landmarks[i,1]));
			l_x = l_x + landmarks[i,0];
			l_y = l_y + landmarks[i,1];
			#l += landmarks(i);
			cnt = cnt + 1;
		l_x = l_x / cnt;
		l_y = l_y / cnt;
		l_p = [l_x, l_y];
		#print("left eye: " + str(l_p[0]) + ", " + str(l_p[1]));

		# Find the center of the right eye by averaging the points around the eye
		cnt = 0;
		r_x  = 0;
		r_y = 0;
		for i in range(42, 47, 1):	
			#print(str(landmarks[i,0]) + ", "+ str(landmarks[i,1]));	
			r_x = r_x + landmarks[i,0];
			r_y = r_y + landmarks[i,1];
			#l += landmarks(i);
			cnt = cnt + 1;
		r_x = r_x / cnt;
		r_y = r_y / cnt;
		r_p = [r_x, r_y];
		#print("right eye: " + str(r_p[0]) + ", " + str(r_p[1]));

    		return l_p, r_p;

	def estimateRollAngle(self, eye_l, eye_r):
		#print("rotate face...");

		eye_direction = (eye_r[0] - eye_l[0], eye_r[1] - eye_l[1]);
		#print("eye direction: " + str(eye_direction[0]) + ", " + str(eye_direction[0]));
	
		rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]));
		#print("rotate: " + str(rotation));

		rotation = math.degrees(rotation);
		#print("rotate: " + str(rotation));
		return rotation;

	def get_landmarks(self, img):
		#self.rects = self.detector(img, 1);
		self.shape = self.predictor(img, self.rects[0]).parts();
		landmarks = np.matrix([[p.x, p.y] for p in self.shape]);
		#print(landmarks);

		return landmarks;

	
	def findFace(self, img):
		ret = True;
		self.rects = self.detector(img, 1);
		if(len(self.rects) == 0):
			print("ERROR: no faces found!");
			ret = False;
		return ret;

	def printFaceDetails(self):
		print("\nDetails of face detection");
		print("-------------------------------");

		print("Number of faces detected: {}".format(len(self.dets)));
		print("box l: " + str(self.box_l) );
		print("box r: " + str(self.box_r) );
		print("box t: " + str(self.box_top) );
		print("box b: " + str(self.box_bottom) );
		
		print("\nlandmarks");
		print("Part 0: {} ...".format(self.shape.part(0)));	
		print("Part 1: {} ...".format(self.shape.part(1)));

		print("-------------------------------\n");
		#print("Processing file: {}".format(img));
		#print("Number of faces detected: {}".format(len(dets)));
		#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()));

	def showDetectedFace(self, img):
		print("show detected face...");

		win = dlib.image_window();

		win.clear_overlay()
    		win.set_image(img)

		#for k, d in enumerate(self.rects):
			# Draw the face landmarks on the screen.
		#win.add_overlay(landmarks);
		
		# draw bounding box
		win.add_overlay(self.rects);
		dlib.hit_enter_to_continue();


