from collections import Counter, OrderedDict
from glob import glob
import os
import sys

import cv2
import numpy
import numpy as np
import pandas as pd
import torch
import h5py
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import pickle as pkl
import albumentations as A
import random
from tqdm import tqdm


class Video_Classification_Dataloader(Dataset):
    def __init__(self, window_path, img_size, loader):

        self.window_path = window_path  # Dataset/images
        self.new_shape = (img_size, img_size) # (112, 112)
        self.loader = loader

        self.idx_to_class_name = {0: 'NoViolence',
                                  1: 'Violence'}

        self.class_name_to_idx = {value: key for key, value in self.idx_to_class_name.items()}

        class_map_file = open(os.getcwd() + os.sep + "class_dict.pkl", "wb+")
        pkl.dump(self.class_name_to_idx, class_map_file)
        class_map_file.close()

        self.num_classes = len(self.idx_to_class_name)


        all_hdf5_paths = [elem for elem in glob(self.window_path + loader + os.sep + "*")]
        print("all_hdf5_paths (before shuffling) =", all_hdf5_paths)
        random.shuffle(all_hdf5_paths)
        print("all_hdf5_paths (after shuffling) =", all_hdf5_paths)

        print("self.window_path + loader =", self.window_path + loader)
        # print("all_hdf5_paths =", all_hdf5_paths)

        self.all_data = []
        self.all_lbls = []
        for idx, elem in tqdm(enumerate(all_hdf5_paths), total=len(all_hdf5_paths)):
            with h5py.File(elem, "r") as hf:
                windows = hf["data"][:]
                labels = hf["labels"][()].decode('utf-8')

                # data = [np.frombuffer(img_bytes, dtype=np.uint8).reshape(16, height, width, channels) for img_bytes in data_bytes]
                resized_window = np.array([np.array([cv2.resize(frame, self.new_shape) for frame in window], dtype=np.uint8) for window in windows], dtype=np.uint8)
                self.all_data.extend(resized_window)
                self.all_lbls.extend([labels] * len(windows))

            """
            if idx > 0.05 * len(all_hdf5_paths):
                break
            """

        print("self.all_lbls =", self.all_lbls)
        print("Counter(self.all_lbls) =", Counter(self.all_lbls))





    def __len__(self):
        return len(self.all_data)



    def __getitem__(self, idx):
        window = torch.FloatTensor(self.all_data[idx])
        window_permuted = window.permute(3, 0, 1, 2)
        lbl = torch.tensor(self.class_name_to_idx[self.all_lbls[idx]])
        return window_permuted, lbl
