from runModel import runModel
import numpy as np


# fix random seed for reproducibility
seed = 7;
np.random.seed(seed);


input_path = "/media/administrator/Working_SSD/Daniel/FaceDB_4fps/";
#input_path = "/media/administrator/Working_SSD/Daniel/ExtendedFeatureDB_4fps/";
output_path = "/media/administrator/Working/Daniel/Output/face_database_20_img/3D_CNN/Dev/";

run = runModel(input_path, output_path);
#run.runTrainingMode();
#run.runAdditionalTrainingMode(output_path);
#run.runHyperparameterSearch(10);
#run.runEvaluateBestModel();
#run.calculateTrainingsPredictions();
#run.runTestMode();
run.run_visualization();

