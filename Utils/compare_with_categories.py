import os
from glob import glob

import numpy as np

root_path = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\"
category = "Fighting"
# category = "Stealing_Fighting"
dir_path = root_path + "Violence_Categories\\" + category + "\\"

all_GT_category_files = [elem for elem in glob(dir_path + "*.mp4")]
all_GT_category_files_sorted = sorted(all_GT_category_files, key=lambda elem: int(elem.split('_')[-1].split('.')[0]))
print("all_GT_category_files_sorted =", all_GT_category_files_sorted)

pred_path = root_path + "Predictions_all\\"
# pred_path = root_path + "Predictions_blurred\\"
correct_pred_files = [elem.split(os.sep)[-1] for elem in glob(pred_path + "Correct_Predicted\\*.mp4")]
wrong_pred_files = [elem.split(os.sep)[-1] for elem in glob(pred_path + "Wrong_Predicted\\*.mp4")]


conf_matrix = np.zeros((2,2))
for file_gt in all_GT_category_files_sorted:
    filename_gt = file_gt.split(os.sep)[-1]
    if filename_gt in correct_pred_files:
        conf_matrix[0,0] += 1
    else:
        conf_matrix[1,0] += 1
        print(filename_gt)

print(conf_matrix)