class ImageDataset:
	# A dataset, consisting of multiple samples/images
	# and corresponding class labels.

	def size(self):
		# Returns the size of the dataset (number of images).
		return 0;


	def nclasses(self):
		# Returns the number of different classes.
		# Class labels start with 0 and are consecutive.
		return 0;


	def classname(self, cid):
		# Returns the name of a class as a string.
		return "";

	def sample(self, sid):
		# Returns the sid-th sample in the dataset, and the
		# corresponding class label. Depending of your language,
		# this can be a Matlab struct, Python tuple or dict, etc.
		# Sample IDs start with 0 and are consecutive.
		# The channel order of samples must be RGB.
		# Throws an error if the sample does not exist.
		return None;
		
