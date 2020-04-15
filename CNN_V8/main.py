from runModel import runModel
import numpy as np

# fix random seed for reproducibility
seed = 7;
np.random.seed(seed);

input_path = "XXXXX/FaceDB_4fps/";
output_path = "XXXXXX/05012017/";

run = runModel(input_path, output_path);
run.runTrainingMode();
#run.runHyperparameterSearch(10);
#run.runEvaluateBestModel();


