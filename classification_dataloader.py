import os
import sys
import glob
import time
import random
from scipy.stats import truncnorm
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
import pickle as pkl
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from collections import Counter
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset

# Imports needed for quantum augmentation:
import scipy
from scipy.linalg import expm
# import Quatum_Augmentation_Operations as QAO
# import Quantum_augmentation_torch_Matthias as QA
import scipy
from scipy import ndimage, misc
from scipy.linalg import expm


class Classification_Dataloader(Dataset):

    def __init__(self, dataset_root_path, loader, cam, background_mode, model_name, win_size, step_size, scaler=None):

        self.loader = loader
        self.dataset_path = dataset_root_path
        self.cam = cam
        self.background_mode = background_mode
        self.model_name = model_name
        self.win_size = win_size
        self.step_size = step_size


        """
        if loader == "train":
            dataset_path = dataset_root_path + "Feature_Matrix_Train.csv"
        elif loader == "val":
            dataset_path = dataset_root_path + "Feature_Matrix_Val.csv"
        elif loader == "test":
            dataset_path = dataset_root_path + "Feature_Matrix_Test.csv"
        else:
            sys.exit("Loader mode " + loader + " is not defined. Please use one of the following modes: train, val, test")
        """


        if loader == "Train":
            self.dataset_sessions = [1, 3, 4, 5, 6] # org
            # self.dataset_sessions = [1, 3, 5, 6, 7]
        elif loader == "Val":
            self.dataset_sessions = [7] # org
            # self.dataset_sessions = [4]
        elif loader == "Test":
            self.dataset_sessions = [2] # org
            # self.dataset_sessions = [2] # org


        print("dataset_path =", self.dataset_path)
        data = self.load_data()
        print(loader + "_data.shape (before filtering) =", data.shape)
        data = self.filter_inconsistency(data)
        print(loader + "_data.shape (after filtering) =", data.shape)



        self.x_data = data[:,:-1]
        self.y_data = data[:,-1]

        if loader == "Train":
            # self.scaler = MinMaxScaler()
            self.scaler = StandardScaler()
            self.x_data = self.scaler.fit_transform(self.x_data)
        else:
            self.scaler = scaler
            self.x_data = scaler.transform(self.x_data)

        self.class_idx_to_name = {0: "walk",
                                  1: "bendover",
                                  2: "stand"
                                  }
        self.class_name_to_idx = {value: key for key, value in self.class_idx_to_name.items()}

        class_map_file = open(os.getcwd() + os.sep + "class_dict.pkl", "wb+")
        pkl.dump(self.class_name_to_idx, class_map_file)
        class_map_file.close()


        self.num_classes = len(self.class_idx_to_name)
        self.class_count_dict = Counter(self.y_data)
        self.num_features = self.x_data.shape[1]
        # print("self.num_features (***) =", self.num_features)
        # print("self.x_data.shape (***) =", self.x_data.shape)
        # print("type(self.x_data[0,0]) (***) =", type(self.x_data[0,0]))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_data(self):
        final_data_array = None
        pbar = tqdm(total=len(self.dataset_sessions) * 3)
        for session in self.dataset_sessions:
            for button_color in ["red", "green", "blue"]:
                file_name = self.background_mode + "_" + self.model_name + "_s" + str(session) + "_cam" + str(self.cam) + "_" + button_color + "_" + str(self.win_size) + "_" + str(self.step_size)
                print("file_name ***", file_name)
                temp_data = np.loadtxt(self.dataset_path + file_name + ".csv")

                if final_data_array is None:
                    final_data_array = temp_data
                else:
                    final_data_array = np.vstack((final_data_array, temp_data))
                pbar.update()

        return final_data_array

    def filter_inconsistency(self, feature_matrix):
        df = pd.DataFrame(feature_matrix)

        # Gruppiere nach den Features und filtere nur die Gruppen, bei denen alle Labels gleich sind
        filtered_groups = df.groupby(list(range(df.shape[1] - 1))).filter(lambda x: x.iloc[:, -1].nunique() == 1)

        # Konvertiere das gefilterte DataFrame zurück zu einer NumPy-Array
        feature_matrix_filtered = filtered_groups.to_numpy()
        return feature_matrix_filtered

    def __len__(self):
        return len(self.y_data)


    def __getitem__(self, idx):
        next_batch = np.array(self.x_data[idx], dtype=np.float32)
        next_lbl = self.y_data[idx]
        return torch.FloatTensor(next_batch), int(next_lbl)
