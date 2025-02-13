class ClassificationDataset:
	# A dataset consisting of multiple samples and corresponding class labels.

	def size(self):
		# Return the size of the dataset (number of samples).
		return 0;

	def nclasses(self):
		# Return the number of different classes.
		# Class labels start with 0 and are consecutive.
		return 0;

	def classname(self, cid):
		# Return the name of a class as a string.
		return "";

	def sample(self, sid):
		# Return the sid-th sample in the dataset, and the
		# corresponding class label. Depending of your language,
		# this can be a Matlab struct, Python tuple or dict, etc.
		# Sample IDs start with 0 and are consecutive.
		# Throws an error if the sample does not exist.
		return None;
		
