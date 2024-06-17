import sys
import os
import random
import numpy as np
import glob
import pickle as pkl
import torch
from hyperparameter_search import HPS
# from __future__ import print_function, division
from video_classification_dataloader import Video_Classification_Dataloader
from train import train_model, test_model, predict_video
from Plot_Conf_Matrix import Plot_Confusion_Matrix

import time
import datetime
import shutil

from tqdm import tqdm

import yaml

# import path_definitions

# date_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S')


# import h5py

# torch.cuda.current_device()
# np.random.seed(0)
# random.seed(0)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':

    train = False  # True, False
    use_those_hyps_once = True
    sbatch_ID = 0  # int(sys.argv[1])

    seed_everything(sbatch_ID)


    # parent_dir = (os.sep).join(os.getcwd().split(os.sep)[:-1]) + os.sep
    yaml_path = os.getcwd() + os.sep + "config.yaml"
    print("yaml path =", yaml_path)

    yaml_file = open(yaml_path, 'r')
    yaml_data = yaml.safe_load(yaml_file)
    # yaml_data = yaml.load(yaml_file)            # yaml_data = yaml.safe_load(yaml_file)
    yaml_file.close()
    print("yaml_data =", yaml_data)

    # data_dir = yaml_data["data_path"]

    model_name = yaml_data["MODEL_NAME"]

    # dataset_root_path, loader, cam, background_mode, win_size, step_size, model_name, scaler=None
    print("yaml_data[WIN_PATH] =", yaml_data["WIN_PATH"])
    print("yaml_data[IMG_SIZE] =", yaml_data["IMG_SIZE"])



    if train:
        train_dataset = Video_Classification_Dataloader(window_path=yaml_data["WIN_PATH"],
                                                        img_size=yaml_data["IMG_SIZE"],
                                                        loader='Train')

        test_dataset = Video_Classification_Dataloader(window_path=yaml_data["WIN_PATH"],
                                                       img_size=yaml_data["IMG_SIZE"],
                                                       loader='Test')

        print("os.getcwd() =", os.getcwd())

        best_evolve_weights_path = os.getcwd() + os.sep + "Checkpoints_RUN" + str(sbatch_ID) + os.sep + "Best_Evolve_Weights" + os.sep

        if not os.path.isdir(best_evolve_weights_path):
            os.makedirs(best_evolve_weights_path)

        use_loss_as_return = False
        use_evolve = True
        if use_evolve:
            evolve_path = os.getcwd() + os.sep + "evolve_RUN" + str(sbatch_ID) + ".txt"

            ######### Evolve Values #########
            num_generations = 10000
            # evolve_path = os.getcwd() + os.sep + "evolve" + date_time + ".txt"

            hps_search = HPS(yaml_data=yaml_data, evolve_path=evolve_path)

            for individum in range(num_generations):
                time_as_int = int(time.time())
                seed_everything(time_as_int)

                are_there_previous_individuals = False
                if os.path.isfile(evolve_path) and len(np.loadtxt(evolve_path)) > 0:
                    are_there_previous_individuals = True
                    all_individuals = np.loadtxt(evolve_path)

                hyps = hps_search.next_generation()

                all_checkpoints = [elem for elem in glob.glob(os.getcwd() + os.sep + "Checkpoints_RUN" + str(sbatch_ID) + os.sep + "*.tar")]

                print("all_checkpoints =", all_checkpoints)
                for elem in all_checkpoints:
                    os.remove(elem)

                print("hyps (after mutation and clipping) =", hyps)


                torch.cuda.empty_cache()
                num_epochs = 5 # 25 -> AdamW
                date_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S')
                """
                if use_those_hyps_once:
                    hyps["batch_size"] = 6
                    hyps["lr_init"] = 0.0016451754
                hyps["optimizer"] = 1
                """
                # hyps["lr_decay_step_size"] = 1
                # hyps["lr_decay_gamma"] = 0.995
                # hyps["class_weight_0"] = 0.33333
                # hyps["class_weight_1"] = 0.33333
                # hyps["class_weight_2"] = 0.33333
                hyps["seed"] = 1
                seed_everything(1)

                new_fitness_value = train_model(model_name=model_name,
                                                train_dataset=train_dataset,
                                                val_dataset=test_dataset,
                                                test_dataset=test_dataset,
                                                hyps=hyps,
                                                date_time=date_time,
                                                num_epochs=num_epochs,
                                                use_loss_as_return=use_loss_as_return,
                                                sbatch_ID=sbatch_ID)
                torch.cuda.empty_cache()

                print("new_fitness_value =", new_fitness_value)


                """
                result_dict_pkl_file = open(os.getcwd() + os.sep + "result_dict[" + date_time + "].pkl", "wb+")
                pkl.dump(result_dict, result_dict_pkl_file)
                result_dict_pkl_file.close()
                """

                """
                ToDo:
                1. Run without Memory Buffer, i.e. Train for S1, Test S1, Train for S2, Test S1, S2, Train for S3, Test S1, S2, S3
                2. Run with infinit Memory Buffer, i.e. Train for S1, Test S1, Train for S1 and S2, Test S1, S2, Train for S1, S2 and S3, Test S1, S2, S3
                """

                torch.cuda.empty_cache()

                save_current_weights = False
                # checkpoint_load_path = os.getcwd() + os.sep + "Checkpoints_RUN" + str(sbatch_ID) + os.sep + "checkpoint_ResNet34.pth.tar"
                checkpoint_load_path = os.getcwd() + os.sep + "Checkpoints_RUN" + str(sbatch_ID) + os.sep + "checkpoint_ResNet34.pth.tar"
                if os.path.isfile(evolve_path) and len(np.loadtxt(evolve_path)) > 0:
                    # all_individuals_temp = np.loadtxt(evolve_path)
                    if len(all_individuals.shape) > 1:
                        cur_best_fitness_value = all_individuals[0, 0]
                    else:
                        cur_best_fitness_value = all_individuals[0]
                    if (use_loss_as_return and new_fitness_value < cur_best_fitness_value) or (
                            not use_loss_as_return and new_fitness_value > cur_best_fitness_value):
                        save_current_weights = True
                elif not os.path.isfile(evolve_path) and os.path.isfile(checkpoint_load_path):
                    save_current_weights = True

                if save_current_weights:
                    shutil.copy2(checkpoint_load_path, best_evolve_weights_path + "checkpoint_ResNet34_" + str(
                        round(new_fitness_value, 8)) + "[" + date_time + "].pth.tar")

                # print("hyps_list_sorted =", hyps_list_sorted)
                # header = ["best_loss"] + [elem[0] for elem in hyps_list_sorted]
                hyps_list_sorted = sorted(hyps.items(), key=lambda elem: elem[0])

                new_individual = np.array([[new_fitness_value] + [hyps[key] for (key, value) in hyps_list_sorted]])

                if are_there_previous_individuals:
                    all_individuals = np.vstack((new_individual, all_individuals))
                    all_individuals = np.array(sorted(all_individuals, key=lambda row: row[0], reverse=not use_loss_as_return))
                else:
                    all_individuals = new_individual.copy()

                np.savetxt(evolve_path, all_individuals, '%14.8g')
    else:
        # """
        checkpoint_path = os.getcwd() + os.sep + "Checkpoints_RUN0" + os.sep + "Best_Evolve_Weights" + os.sep + "checkpoint_ResNet34_0.91453528[2024-04-03 19_40_09].pth.tar"
        # video_path = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\Violence\\VIOLENCE_548.mp4"
        # video_dir = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\NoViolence\\"
        # video_dir_all = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\"
        video_dir_all = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\Blurred Videos\\"
        # video_dir = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\Violence_Categories\\Fighting\\"
        # save_dir = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\Predictions_category_fighting\\"
        save_dir = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\Predicted_Violence_blurred\\"
        # save_dir = "\\\\169.254.213.20\\NAS_Data\\Datasets\\SDIL (Bus-Videos)\\Predicted_NoViolence_blurred\\"
        correct_predicted = save_dir + "Correct_Predicted" + os.sep
        wrong_predicted = save_dir + "Wrong_Predicted" + os.sep

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if not os.path.isdir(correct_predicted):
            os.makedirs(correct_predicted)

        if not os.path.isdir(wrong_predicted):
            os.makedirs(wrong_predicted)

        # all_video_paths = [elem for elem in glob.glob(video_dir_all + "*" + os.sep + "*.mp4") if elem.split(os.sep)[-2] == "NoViolence" or elem.split(os.sep)[-2] == "Violence"]
        all_video_paths = [elem for elem in glob.glob(video_dir_all + "*" + os.sep + "*.mp4") if elem.split(os.sep)[-2] == "NoViolence_blurred" or elem.split(os.sep)[-2] == "Violence_blurred"]
        # all_video_paths = [elem for elem in glob.glob(video_dir + "*.mp4")]
        random.shuffle(all_video_paths)
        # """
        idx_to_class_name = {0: 'NoViolence',
                             1: 'Violence'}

        class_name_to_idx = {v: k for k, v in idx_to_class_name.items()}
        # """
        conf_matrix = np.zeros((len(idx_to_class_name),len(idx_to_class_name)))

        for video_path in all_video_paths:
            video_name = video_path.split(os.sep)[-1]
            print("video name:", video_name)
            preds_dict = predict_video(model_name=model_name,
                                       checkpoint_path=checkpoint_path,
                                       video_path=video_path,
                                       save_dir=save_dir)

            lbl_classes = [preds_dict["lbl_cls_ID"]] * preds_dict["num_windows"]
            at_least_one_win_wrong_pred = False
            for pred_cls, lbl_cls in zip(preds_dict["classes_per_win"], lbl_classes):
                conf_matrix[int(pred_cls), int(lbl_cls)] += 1
                if pred_cls != lbl_cls:
                    at_least_one_win_wrong_pred = True

            if at_least_one_win_wrong_pred:
                shutil.move(src=save_dir + video_name, dst=wrong_predicted + video_name)
            else:
                shutil.move(src=save_dir + video_name, dst=correct_predicted + video_name)
        # """
        """
        conf_matrix = np.array([[2293.,   21.],
                                [462., 2725.]])
        
        conf_matrix = np.array([[510.,  95.],
                                [56., 439.]])
        """
        date_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S')
        average_Precision, average_Recall, average_F1 = Plot_Confusion_Matrix.plot_confusion_matrix(conf_matrix=conf_matrix,
                                                                                                    save_path=os.getcwd() + os.sep + "conf_matrix[" + date_time + "]",
                                                                                                    # save_path=os.getcwd() + os.sep + "conf_matrix",
                                                                                                    idx_to_class_name=idx_to_class_name,
                                                                                                    missing_classes=[])

