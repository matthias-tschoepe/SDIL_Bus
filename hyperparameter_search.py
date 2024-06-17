import sys
import os
import random
import numpy as np
import glob
import pickle as pkl
import torch


import time
import datetime
import shutil

from tqdm import tqdm

import yaml

class HPS:
    def __init__(self, yaml_data, evolve_path, use_loss_as_return=False, sbatch_ID=0):
        self.yaml_data = yaml_data
        self.sbatch_ID = sbatch_ID
        self.max_num_individuals_for_evolve = 10  # len(all_individuals) # 10
        self.use_loss_as_return = use_loss_as_return
        self.are_there_previous_individuals = False

        self.evolve_path = evolve_path
        print("evolve_path =", self.evolve_path)

        # Copy Start Values once
        if not os.path.isfile(self.evolve_path) and os.path.isfile(os.getcwd() + os.sep + "evolve.txt"):
            shutil.copy2(os.getcwd() + os.sep + "evolve.txt", self.evolve_path)

        self.hyps = {"alpha_RMS": 0,
                     "amsgrad": 0,  # True or False
                     "batch_size": 4,  # 16 32,
                     "beta_1": 5.3068784e-05,
                     "beta_2": 0.010985558,
                     "centered": 0,  # True or False
                     "dampening": 0,
                     "eps": 1.0342923e-08,
                     "lr_decay_Adagrad": 0,
                     "lr_decay_gamma": 0.99995,  # 0.95,
                     "lr_decay_step_size": 1,  # 5,
                     "lr_init": 4.5625572e-05,
                     "momentum":2.4893828e-05,
                     "nesterov": 0,  # True or False
                     "optimizer": 2,  # Integer in range [0,4]
                     "pretrained_weights": 1,
                     "seed": 1,
                     "weight_decay": 0.0001467172
                    }

        self.hyps_bounds = {"alpha_RMS": (0, 1),
                            "amsgrad": (0, 1),  # True or False
                            "batch_size": (4, 8),  # (2, 16) (2, 1000); (2, 300)
                            "beta_1": (0, 0.999999),
                            "beta_2": (0, 0.999999),
                            "centered": (0, 1),  # True or False
                            "dampening": (0, 1),
                            "eps": (0, 1),
                            "lr_decay_Adagrad": (0, 1),
                            "lr_decay_gamma": (0.0001, 1),
                            "lr_decay_step_size": (1, 50),
                            "lr_init": (0.00000001, 0.9),  # 0.00001
                            "nesterov": (0, 1),  # True or False
                            "momentum": (0, 1),
                            "optimizer": (0, 4),  # Integer in range [0,4]
                            "pretrained_weights": (0, 1),
                            "seed": (1, 9999),  # (1, 200)
                            "weight_decay": (0, 1)
                           }

        self.hyps_variance = {"alpha_RMS": 1.0,
                              "amsgrad": 1.0,  # True or False
                              "batch_size": 1.0,  # 16 32,
                              "beta_1": 1.0,
                              "beta_2": 1.0,
                              "centered": 1.0,  # True or False
                              "dampening": 1.0,
                              "eps": 1.0,
                              "lr_decay_Adagrad": 1.0,
                              "lr_decay_gamma": 1.0,  # 0.95,
                              "lr_decay_step_size": 1.0,  # 5,
                              "lr_init": 1.0,
                              "momentum": 1.0,
                              "nesterov": 1.0,  # True or False
                              "optimizer": 1.0,  # Integer in range [0,4]
                              "pretrained_weights": 1.0,
                              "seed": 1.19,
                              "weight_decay": 1.0
                             }



        if os.path.isfile(self.evolve_path):
            evolve_data = np.loadtxt(self.evolve_path)
            if len(evolve_data.shape) > 1:
                best_evolve_entry = list(evolve_data[0][1:])
                for i, (key, value) in enumerate(self.hyps.items()):
                    self.hyps[key] = best_evolve_entry[i]

                    if key in ["batch_size", "seed"]:
                        self.hyps[key] = int(self.hyps[key])
            else:
                best_evolve_entry = list(evolve_data[1:])
        print("*** self.hyps *** =", self.hyps)

        self.keys_sorted = sorted(self.hyps.keys())
        print("self.keys_sorted =", self.keys_sorted)
        self.hyps_list_sorted = sorted(self.hyps.items(), key=lambda elem: elem[0])
        print("self.hyps_list_sorted =", self.hyps_list_sorted)


    def next_generation(self):
        time_as_int = int(time.time())
        self.seed_everything(time_as_int)
        # np.random.seed(self.hyps["seed_numpy"])
        # torch.manual_seed(self.hyps["seed_torch"])


        # Wenn die evolve Datei existiert und Daten enthält, dann können wir aus vorherigen Individuen ein neues Individum berechnen:
        if os.path.isfile(self.evolve_path) and len(np.loadtxt(self.evolve_path)) > 0:
            self.are_there_previous_individuals = True
            all_individuals = np.loadtxt(self.evolve_path)
            print("all_individuals.shape =", all_individuals.shape)
            print("len(all_individuals.shape) =", len(all_individuals.shape))

            if len(all_individuals.shape) > 1:
                num_cols = len(all_individuals[0])

                print("all_individuals (before removing duplicates) =", all_individuals)
                individuals_removed_duplicates = []
                for row_i in all_individuals:
                    add_to_list = True
                    for row_j in individuals_removed_duplicates:
                        if (row_i[1:] == row_j[1:]).all():  # or (row_i[0] == row_j[0]):
                            add_to_list = False
                            break
                    if add_to_list:
                        individuals_removed_duplicates.append(list(row_i))
                all_individuals = np.array(individuals_removed_duplicates).reshape((-1, num_cols)).copy()
                np.savetxt(self.evolve_path, all_individuals, '%14.8g')
                print("all_individuals (after removing duplicates) =", all_individuals)
            else:
                num_cols = len(all_individuals)


            if len(all_individuals.shape) > 1:
                all_individuals = np.array(sorted(all_individuals, key=lambda row: row[0], reverse=not self.use_loss_as_return))
                print("all_individuals =", all_individuals)
                num_individuals_for_evolve = min(self.max_num_individuals_for_evolve, len(all_individuals))
                print("num_individuals_for_evolve =", num_individuals_for_evolve)
                worst_fitness_value = all_individuals[num_individuals_for_evolve - 1, 0]
                w = (1.0 - (all_individuals[:num_individuals_for_evolve,
                            0] / worst_fitness_value)) + 0.00000000001
                print("w =", w)
                w = w / w.sum()
                w_sum = w.sum()
                print("w =", w)
                print("w_sum =", w_sum)

                for j, (key, value) in enumerate(self.hyps_list_sorted):
                    hyp_temp = 0
                    for i in range(num_individuals_for_evolve):
                        hyp_temp += w[i] * value
                    self.hyps[key] = hyp_temp  # / w_sum
            print("new_hyps (before mutation and clipping) =", self.hyps)

            # ToDo: Mutation on self.hyps and boundary check
            mutation_threshold = 0.75
            random_num = random.uniform(0, 1)
            if random_num < 0.33:
                sigma = 0.25  # 0.85
            elif 0.33 <= random_num < 0.66:
                sigma = 0.5  # 0.35, 0.65
            else:
                sigma = 0.75  # 0.35, 0.65

            scale = 1.0  # 0.85 # 1.0
            prob_to_choose_new_seeds = 1.5
            for key, value in self.hyps.items():
                # if np.random.uniform(low=0, high=1) <= mutation_threshold:
                use_genetic_approach = True # np.random.uniform(0,1) < 0.5
                if use_genetic_approach:
                    scale = 1.0  # 0.85 # 1.0
                    # self.hyps[key] = self.hyps[key] * (1 + np.random.normal(loc=0.0, scale=scale) * sigma * self.hyps_variance[key])**2
                    self.hyps[key] = self.hyps[key] * (
                            1 + random.normalvariate(mu=0.0, sigma=scale) * sigma * self.hyps_variance[key]) ** 2
                else:
                    # A simple random search, if the above genetic approach doesn't work well.
                    self.hyps[key] = random.uniform(self.hyps_bounds[key][0], self.hyps_bounds[key][1])

                if key == "batch_size" and random.uniform(0, 1) <= 0.5:
                    self.hyps[key] = int(time.time() % self.hyps_bounds[key][1])

                if key == "seed" and random.uniform(0, 1) <= prob_to_choose_new_seeds:
                    self.hyps[key] = int(time.time() * 17 % self.hyps_bounds[key][1])

                if key in ["batch_size", "lr_decay_step_size", "optimizer", "seed"]:
                    print("self.hyps[" + key + "] =", self.hyps[key])
                    print("round(self.hyps[" + key + "],0) =", round(self.hyps[key], 0))
                    print("int(round(self.hyps[" + key + "],0)) =", int(round(self.hyps[key], 0)))
                    self.hyps[key] = int(round(self.hyps[key], 0))


                if key in ["amsgrad", "centered", "nesterov", "pretrained_weights"]:
                    if self.hyps[key] < 0.5:
                        self.hyps[key] = 0
                    else:
                        self.hyps[key] = 1

                self.hyps[key] = np.clip(self.hyps[key], a_min=self.hyps_bounds[key][0],
                                         a_max=self.hyps_bounds[key][1])

            """
            Handle some Hyperparameter dependencies. For example, there is the following error message if nesterov = Ture and dampening > 0:
                ValueError: Nesterov momentum requires a momentum and zero dampening
            """
            if self.hyps["nesterov"] == 1:
                self.hyps["dampening"] = 0


        self.hyps["batch_size"] = int(self.hyps["batch_size"])


        print("new_hyps (after mutation and clipping) =", self.hyps)

        # average_Precision, average_Recall, average_F1 = 0,0,0
        # self.hyps["eps"] = 1e-08
        self.hyps["optimizer"] = 2  # 0, 2
        self.hyps["pretrained_weights"] = 0

        # self.hyps["lr_decay_step_size"] = 1
        # self.hyps["lr_decay_gamma"] = 0.995
        self.hyps["seed"] = 1

        return self.hyps



    def get_hyps(self):
        return self.hyps

    def load_best_hyps(self):
        evolve_data = np.loadtxt(os.getcwd() + os.sep + "evolve_RUN0.txt")[0][1:]
        keys_sorted = sorted(self.hyps.keys())
        for i, key in enumerate(keys_sorted):
            self.hyps[key] = evolve_data[i]

            if key in ["amsgrad", "centered", "nesterov", "pretrained_weights"]:
                if self.hyps[key] < 0.5:
                    self.hyps[key] = 0
                else:
                    self.hyps[key] = 1

            if key in ["batch_size", "lr_decay_step_size", "optimizer", "seed"]:
                self.hyps[key] = int(round(self.hyps[key], 0))


        return self.hyps



    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



