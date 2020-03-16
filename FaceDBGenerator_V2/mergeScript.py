import numpy as np
import h5py as hf
#hist_equ_database_part1
path = "/media/administrator/Working/Daniel/PreprocessedInput/FaceDatabase/FaceDB_2_FPS/original/";
output_path = "/media/administrator/Working/Daniel/PreprocessedInput/FaceDatabase/FaceDB_2_FPS/";

database_part1 = hf.File(path + "new_database_part1.h5", 'r');
ids = np.array(database_part1["id"][:]);
numberOfImages_part1 = ids.shape[0];
print("part1 number of images: " + str(numberOfImages_part1));

database_part2 = hf.File(path + "new_database_part2.h5", 'r');
ids = np.array(database_part2["id"][:]);
numberOfImages_part2 = ids.shape[0];
print("part2 number of images: " + str(numberOfImages_part2));

database_part3 = hf.File(path + "new_database_part3.h5", 'r');
ids = np.array(database_part3["id"][:]);
numberOfImages_part3 = ids.shape[0];
print("part3 number of images: " + str(numberOfImages_part3));

database_part4 = hf.File(path + "new_database_part4.h5", 'r');
ids = np.array(database_part4["id"][:]);
numberOfImages_part4 = ids.shape[0];
print("part4 number of images: " + str(numberOfImages_part4));

database_part5 = hf.File(path + "new_database_part5.h5", 'r');
ids = np.array(database_part5["id"][:]);
numberOfImages_part5 = ids.shape[0];
print("part5 number of images: " + str(numberOfImages_part5));

database_part6 = hf.File(path + "new_database_part6.h5", 'r');
ids = np.array(database_part6["id"][:]);
numberOfImages_part6 = ids.shape[0];
print("part6 number of images: " + str(numberOfImages_part6));

completeNimages = numberOfImages_part1 + numberOfImages_part2 + numberOfImages_part3 + numberOfImages_part4 + numberOfImages_part5 + numberOfImages_part6;
print("complete dataset number of images: " + str(completeNimages));

# create complete dataset
database_complete = hf.File(output_path + "database_complete.h5", 'a');
database_complete.create_dataset("id", (completeNimages,), dtype='int32');
database_complete.create_dataset("name", (completeNimages,), dtype='S30');
database_complete.create_dataset("data", (completeNimages,64,64,3), dtype='uint8');
database_complete.create_dataset("labels", (completeNimages,5), dtype='float32');


# copy contents to the new database 
end1 = numberOfImages_part1;
database_complete["id"][:end1] = database_part1["id"][:];
database_complete["name"][:end1] = database_part1["name"][:];
database_complete["data"][:end1] = database_part1["data"][:];
database_complete["labels"][:end1] = database_part1["labels"][:];

end2 = (numberOfImages_part1 + numberOfImages_part2);
database_complete["id"][end1:end2] = database_part2["id"][:] + 1200;
database_complete["name"][end1:end2] = database_part2["name"][:];
database_complete["data"][end1:end2] = database_part2["data"][:];
database_complete["labels"][end1:end2] = database_part2["labels"][:];

end3 = (end2 + numberOfImages_part3);
database_complete["id"][end2:end3] = database_part3["id"][:] + 2160;
database_complete["name"][end2:end3] = database_part3["name"][:];
database_complete["data"][end2:end3] = database_part3["data"][:];
database_complete["labels"][end2:end3] = database_part3["labels"][:];

end4 = (end3 + numberOfImages_part4);
database_complete["id"][end3:end4] = database_part4["id"][:] + 3120;
database_complete["name"][end3:end4] = database_part4["name"][:];
database_complete["data"][end3:end4] = database_part4["data"][:];
database_complete["labels"][end3:end4] = database_part4["labels"][:];

end5 = (end4 + numberOfImages_part5);
database_complete["id"][end4:end5] = database_part5["id"][:] + 4080;
database_complete["name"][end4:end5] = database_part5["name"][:];
database_complete["data"][end4:end5] = database_part5["data"][:];
database_complete["labels"][end4:end5] = database_part5["labels"][:];

end6 = (end5 + numberOfImages_part6);
database_complete["id"][end5:] = database_part6["id"][:] + 5040;
database_complete["name"][end5:] = database_part6["name"][:];
database_complete["data"][end5:] = database_part6["data"][:];
database_complete["labels"][end5:] = database_part6["labels"][:];


