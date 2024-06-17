import os
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from Plot_Conf_Matrix import Plot_Confusion_Matrix
from tqdm import tqdm

from PIL import Image

from sklearn.metrics import f1_score as F1_Score

from models import Video_Classification_Models

# from torch.utils.tensorboard import SummaryWriter

# from Fast_SCNN.utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss


import cv2

# For calculating the runtime
import time

# To get the time in date format (for saving the checkpoints)
import datetime

import copy
import sys

# For saving the SVM Feature-Matrix
import pickle as pkl

# For saving the model weights:
import os.path
# from deeplabv3_ResNet18_model_from_github import deeplabv3 as git_deeplabv3


# torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_class_weights(class_count_dict):
    print("class_count_dict =", class_count_dict)
    print("class_count_dict.items() =", class_count_dict.items())
    sorted_class_dict = sorted(class_count_dict.items(), key=lambda elem: elem[0])
    print("sorted_class_dict =", sorted_class_dict)

    power = 2.0
    inverted_values = []
    for key, value in sorted_class_dict:
        temp = (1.0 / float(value)) ** power
        inverted_values.append(temp)

    denominator_sum = np.array(inverted_values).sum()
    class_weigts = []
    print("denominator_sum =", denominator_sum)

    for elem in inverted_values:
        temp = np.float32(float(elem) / denominator_sum)
        class_weigts.append(temp)

    print("class_weigts =", class_weigts)

    return torch.tensor(class_weigts).to(device)


def train_model(model_name, train_dataset, val_dataset, test_dataset, hyps, date_time, num_epochs=100, use_loss_as_return=True, sbatch_ID=None):
    seed_everything(hyps["seed"])

    num_workers = 4 # 120 # 68 # 24
    # temp_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False, drop_last=True)

    # data = next(iter(temp_train_dataloader))
    # print("data[0].shape =", data[0].shape)
    # mean, std = data[0].mean(), data[0].std()
    # print("mean, std (1) =", mean, std)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hyps["batch_size"], pin_memory=True,
                                                   num_workers=num_workers, shuffle=True, drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=hyps["batch_size"], pin_memory=True,
                                                 num_workers=num_workers, shuffle=False, drop_last=False)

    dataset_sizes = dict()
    dataset_sizes["train"] = len(train_dataset)
    dataset_sizes["val"] = len(val_dataset)

    log_dir = os.getcwd() + "/LogFiles_RUN" + str(sbatch_ID) + "/"
    if not (os.path.exists(log_dir)):
        os.makedirs(log_dir)

    # Write Header of Log_File:
    for phase in {'train', 'val'}:
        file = open(log_dir + phase + "_log_[" + date_time + "]_.csv", "w+")
        file.write("Epoch     Loss     Accuracy     F1     Top-1\n")
        file.close()

    # class_count_dict = train_dataset.class_count_dict
    num_classes = train_dataset.num_classes
    # print("class_count_dict =", class_count_dict)
    print("num_classes =", num_classes)

    since = time.time()

    # pretrained = True if hyps["pretrained_weights"] == 1 else False
    # pretrained_weights_flag = ResNet50_Weights.IMAGENET1K_V1 if hyps["pretrained_weights"] == 1 else None
    # print("Using pretrained weights:", pretrained_weights_flag)

    # model = resnet50(pretrained=hyps["pretrained_weights"], num_classes=len(class_count_dict))
    model = Video_Classification_Models(model_name=model_name, num_classes=num_classes)

    model = model.to(device)

    print("Num samples =", len(train_dataset))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = model.cuda()
    """
    use_precomputed_class_weights = True
    if use_precomputed_class_weights:
        # class_weights = torch.tensor([0.3333, 0.3333, 0.3333, 0.0001]).to(device)  # for all 243 labeled standard Images
        class_weights = compute_class_weights(class_count_dict)
    else:
        class_weights = torch.tensor([np.float32(hyps["class_weight_0"]), np.float32(hyps["class_weight_1"]), np.float32(hyps["class_weight_2"])]).to(device)
    print("class_weights =", class_weights)
    criterion = nn.CrossEntropyLoss(class_weights)
    """
    criterion = nn.CrossEntropyLoss()

    """
    use_Fast_SCNN_Loss = True
    if use_Fast_SCNN_Loss:
        criterion = MixSoftmaxCrossEntropyOHEMLoss()
    """

    use_individual_decay = False

    # cur_lr = 0.01 # 0.0001 # Adam
    # cur_lr = 0.0001

    # Observe that all parameters are being optimized
    # optimizer = torch.optim.SGD(model.parameters(), lr=cur_lr, momentum=0.82, weight_decay=0.5, nesterov=True) #, weight_decay=0.0005)  # my default optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=cur_lr, momentum=0.92, weight_decay=0.0005, nesterov=True) # used for DeepLabv3 and U-Net in Master Thesis
    # optimizer = torch.optim.SGD(model.parameters(), lr=cur_lr, momentum=0.9, weight_decay=0.00004) # Fast SCNN in Master Thesis
    # optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr) #, weight_decay=0.0005)    # diverges for 4 classes
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cur_lr, betas=(0.6, 0.9), weight_decay=0.00085)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cur_lr, weight_decay=0.00085)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cur_lr, betas=(0.6, 0.9), weight_decay=0.00085)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)

    # best_model_wts = copy.deepcopy(model.state_dict())
    last_loss = 10 ** 20

    epoch = 0

    checkpoint_save_path_1 = os.getcwd() + "/Checkpoints_RUN" + str(sbatch_ID) + "/checkpoint_[" + date_time + "]_ResNet34.pth.tar"
    checkpoint_save_path_2 = os.getcwd() + "/Checkpoints_RUN" + str(sbatch_ID) + "/checkpoint_ResNet34.pth.tar"
    # checkpoint_load_path = os.getcwd() + "/Pretrained_Weights/checkpoint_ResNet34.pth.tar"
    checkpoint_load_path = checkpoint_save_path_2

    print("checkpoint_load_path =", checkpoint_load_path)

    if os.path.isfile(checkpoint_load_path) and model is None:
        print("=> Looking for checkpoint")
        try:
            print("=> Looking for checkpoint")
            checkpoint = torch.load(checkpoint_load_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> Found and loaded checkpoint")
            epoch = checkpoint['epoch']
            print("Loaded epoch number is", epoch)
            epoch = 0
            print("Reset epoch number to", epoch)
            last_loss = checkpoint['best_loss']
            print("Loaded best loss is", last_loss)
            # optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print("Your checkpoint does not contain trained weights. Your old weights will be overwritten.")
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, checkpoint_save_path_1)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, checkpoint_save_path_2)
        # print("The loaded model ends in the " + str(epoch) + " epoch, with a validation accuracy of: {:.4f}".format(best_acc.item()))
    else:
        print("=> No checkpoint found. You have to train first.")
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, checkpoint_save_path_1)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, checkpoint_save_path_2)

    cur_lr = hyps["lr_init"]
    momentum = hyps["momentum"]
    dampening = hyps["dampening"]
    weight_decay = hyps["weight_decay"]
    nesterov = True if hyps["nesterov"] == 1 else False
    betas = (hyps["beta_1"], hyps["beta_2"])
    eps = hyps["eps"]
    amsgrad = True if hyps["amsgrad"] == 1 else False
    lr_decay_Adagrad = hyps["lr_decay_Adagrad"]
    alpha = hyps["alpha_RMS"]
    centered = True if hyps["centered"] == 1 else False

    if hyps["optimizer"] == 0:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cur_lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    dampening=dampening,
                                    nesterov=nesterov)
    elif hyps["optimizer"] == 1:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cur_lr,
                                     betas=betas,
                                     eps=eps,
                                     weight_decay=weight_decay,
                                     amsgrad=amsgrad)
    elif hyps["optimizer"] == 2:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cur_lr,
                                      betas=betas,
                                      eps=eps,
                                      weight_decay=weight_decay,
                                      amsgrad=amsgrad)
    elif hyps["optimizer"] == 3:
        optimizer = torch.optim.Adagrad(model.parameters(),
                                        lr=cur_lr,
                                        lr_decay=lr_decay_Adagrad,
                                        weight_decay=weight_decay, eps=eps)
    elif hyps["optimizer"] == 4:
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=cur_lr,
                                        momentum=momentum,
                                        alpha=alpha,
                                        eps=eps,
                                        centered=centered,
                                        weight_decay=weight_decay)

    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.316)       # Fast-SCNN
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)       # Deeplabv3 and U-Net
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=hyps["lr_decay_step_size"],
                                    gamma=hyps["lr_decay_gamma"])  # Deeplabv3 and U-Net
    # milestones_list = [15,50,100,200,300,400,500,600,800,900]
    # milestones_list = [100,120,150]
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.1)



    better_val_score_found = False
   
    while epoch <= num_epochs:
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        val_img_counter = 1

        for param_group in optimizer.param_groups:
            if use_individual_decay:
                poly_decay = (1.0 - (float(epoch) / float(num_epochs))) ** 0.9
                param_group['lr'] = hyps['lr_init'] * poly_decay
            # cur_lr = param_group['lr']
            print("Current learning rate:", param_group['lr'])

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_dataloader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_dataloader
            running_loss = 0.0
            
            total_correct = 0.0
            total_examples = 0.0
            total_f1_score = 0.0
            total_top1_correct = 0.0

            running_corrects = 0

            counter = 0
            # Iterate over data.
            # for elem in dataloaders[phase]:
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))  # progress bar
            for idx, (inputs, labels) in pbar:
                counter += 1
                # inputs = inputs.float()       # maybe necessary, was in Davids Code
                inputs = inputs.to(device)
                labels = labels.to(device)


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.float()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # For average Loss
                    running_loss += loss.item() * labels.size(0)

                    # y_pred_softmax = torch.softmax(outputs, dim=1)
                    y_pred_softmax = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(y_pred_softmax, 1)
                    # loss = criterion(outputs, torch.max(labels, 1)[1])

                    # For average Accuracy
                    batch_correct = (preds == labels).sum().item()
                    total_correct += batch_correct
                    total_examples += labels.size(0)
                    
                    # For average F1-Score
                    batch_f1_score = F1_Score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    total_f1_score += batch_f1_score
                    
                    # For average Top-1 Score
                    _, batch_top1_predicted = torch.topk(outputs, 1)
                    batch_top1_correct = torch.sum(batch_top1_predicted == labels.view(-1, 1).expand_as(batch_top1_predicted)).item()
                    total_top1_correct += batch_top1_correct


                """
                if idx > 25:
                    break
                """

            if dataset_sizes[phase] > 0:
                # epoch_loss = running_loss / dataset_sizes[phase]
                
                epoch_loss = running_loss / total_examples
                
                accuracy = total_correct / total_examples
                f1_score = total_f1_score / len(dataloader)
                top1_accuracy = total_top1_correct / total_examples

            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('\n{} Loss: {:.4f}'.format(phase, epoch_loss))
            print(phase, "Loss:", epoch_loss)

            # Write Loss and Accuracy:
            file = open(log_dir + phase + "_log_[" + date_time + "]_.csv", "a+")
            file.write(str(epoch) + "     " + str(epoch_loss) + "     " + str(accuracy) + "     " + str(f1_score) + "     " + str(top1_accuracy) + "\n")
            file.close()

            # If the current Validation Accuracy is better than the previous one, than save the new model-weights:
            if phase == 'val' and epoch_loss < last_loss:
                last_loss = epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                print("\t--------- ********************************* ---------")
                print("\t--------- *** New best Validation score *** ---------")
                print("\t--------- ********************************* ---------")
                
                better_val_score_found = True
                sys.stdout.write("Saving model weights. Do not interrupt the saving-process ...")
                sys.stdout.flush()
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_loss': last_loss,
                            'optimizer': optimizer.state_dict()}, checkpoint_save_path_1)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_loss': last_loss,
                            'optimizer': optimizer.state_dict()}, checkpoint_save_path_2)
                sys.stdout.write(" done.\n")
                sys.stdout.flush()
            
            
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_loss': last_loss, 'optimizer': optimizer.state_dict()}, checkpoint_save_path_1)
            
            """
            if epoch_loss > last_loss:
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > 0.0001:
                        param_group['lr'] = param_group['lr'] * 0.5
            """

            """
            if epoch == num_epochs:
                # best_model_wts = copy.deepcopy(model.state_dict())
                print("\t--------- ****************************** ---------")
                print("\t--------- *** Reached the last epoch *** ---------")
                print("\t--------- ****************************** ---------")

                sys.stdout.write("Saving model weights. Do not interrupt the saving-process ...")
                sys.stdout.flush()
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_loss': epoch_loss, 'optimizer': optimizer.state_dict()}, checkpoint_save_path_1)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_loss': epoch_loss, 'optimizer': optimizer.state_dict()}, checkpoint_save_path_2)
                sys.stdout.write(" done.\n")
                sys.stdout.flush()
            """

        """
        if cur_lr > 0.00001:
            scheduler.step(epoch)
        """
        if not use_individual_decay:
            scheduler.step()

        # scheduler.step(3)
        epoch += 1
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss Acc: {:4f}'.format(last_loss))

    
    # checkpoint = torch.load(checkpoint_load_path)
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if better_val_score_found:
        checkpoint = torch.load(checkpoint_save_path_2)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        checkpoint = torch.load(checkpoint_save_path_1)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    

    if use_loss_as_return:
        # load best model weights
        # model.load_state_dict(best_model_wts)
        return last_loss, model
    else:
        # load best model weights
        # model.load_state_dict(best_model_wts)
        # return last_loss
        average_Precision, average_Recall, average_F1 = test_model(model_name=model_name,
                                                                   test_dataset=test_dataset,
                                                                   date_time=date_time,
                                                                   checkpoint_path=None,
                                                                   trained_model=model)
        return average_F1




def test_model(model_name, test_dataset, date_time, checkpoint_path=None, trained_model=None):
    if trained_model is None:
        print("test_dataset.num_classes =", test_dataset.num_classes)
        model = Video_Classification_Models(model_name=model_name, num_classes=test_dataset.num_classes)
    else:
        model = trained_model

    model = model.to(device)
    model.eval()

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  num_workers=1,
                                                  shuffle=False,
                                                  drop_last=False)

    num_classes = test_dataset.num_classes

    if trained_model is None or trained_model == False:
        if os.path.isfile(checkpoint_path):
            print("=> Looking for checkpoint")
            checkpoint = torch.load(checkpoint_path)
            epoch = checkpoint['epoch']
            print("Loaded epoch number is", epoch)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> Found and loaded checkpoint")
        else:
            print("=> No checkpoint found. You have to train first.")
            sys.exit()

    conf_matrix = np.zeros((num_classes, num_classes))
    dummy_counter = 0
    num_correct_predicted = 0
    num_wrong_predicted = 0

    for inputs, labels in tqdm(test_dataloader):
        """
        if dummy_counter > 5000:
            break
        dummy_counter += 1
        """
        # print("inputs (after) =", inputs)
        # inputs = inputs.float()  # maybe necessary, was in Davids Code
        inputs = inputs.to(device)

        outputs = model(inputs)
        # _, preds = torch.max(outputs, 1)

        y_pred_softmax = torch.nn.functional.softmax(outputs, dim=1)

        # pred = output[0].permute(1, 2, 0).detach().cpu().numpy()
        _, pred_idx = torch.max(y_pred_softmax, 1)
        # pred_cls = class_map_idx_to_lbl[float(pred_idx)]
        pred_cls = int(pred_idx)
        # print("pred_cls =", pred_cls)
        conf_matrix[pred_cls, labels] += 1

        if pred_cls == labels:
            num_correct_predicted += 1
        else:
            num_wrong_predicted += 1

        """
        print("labels_one_hot =", labels_one_hot)
        print("label_cls =", label_cls)
        print("pred =", pred)
        print("pred_idx =", pred_idx)
        print("pred_cls =", pred_cls)
        """

        # print("num_correct_predicted =", num_correct_predicted)
        # print("num_wrong_predicted =", num_wrong_predicted)

    # np.set_printoptions(precision=4, suppress=True)
    # print(conf_matrix)
    """
    conf_mat_pkl_file = open(os.getcwd() + os.sep + "conf_matrix[" + date_time + "].pkl", "wb+")
    pkl.dump(conf_matrix, conf_mat_pkl_file)
    conf_mat_pkl_file.close()
    """

    class_map_file = open(os.getcwd() + os.sep + "class_dict.pkl", "rb")
    class_map_to_string = pkl.load(class_map_file)
    class_map_file.close()
    print("class_map_to_string =", class_map_to_string)

    # class_map_to_string = test_dataset.class_map_to_string
    conf_matrix = [conf_matrix, os.getcwd() + os.sep + "conf_matrix[" + date_time + "].pdf"]
    print("conf_matrix[1] (save path) =", conf_matrix[1])

    average_Precision, average_Recall, average_F1 = Plot_Confusion_Matrix.plot_confusion_matrix(conf_matrix,
                                                                                                idx_to_class_name=class_map_to_string,
                                                                                                missing_classes=[],
                                                                                                plot_conf_matrix=False)
    # average_Precision, average_Recall, average_F1
    print("average_Precision, average_Recall, average_F1 =", average_Precision, average_Recall, average_F1)
    return average_Precision, average_Recall, average_F1



def embedd_img_in_gray_background(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    resize_large_height = 256
    resize_large_width = 256

    img_as_array = np.array(pil_img)
    largest_side_of_image = max(img_as_array.shape[0], img_as_array.shape[1])
    largest_side_of_embedding = max(resize_large_height, resize_large_width)
    resize_ratio = largest_side_of_embedding / largest_side_of_image

    pil_img = pil_img.resize((int(img_as_array.shape[1] * resize_ratio),
                      int(img_as_array.shape[0] * resize_ratio)), Image.ANTIALIAS)

    img_as_array = np.array(pil_img)

    img_height, img_width, _ = img_as_array.shape

    embedding = np.ones((resize_large_height, resize_large_width, 3), dtype=np.uint8) + 127

    width_start = int(embedding.shape[1] / 2) - int(img_width / 2)
    width_end = width_start + img_width
    height_start = int(embedding.shape[0] / 2) - int(img_height / 2)
    height_end = height_start + img_height

    # print("width_end-width_start =", (width_end-width_start))
    # print("height_end-height_start =", (height_end-height_start))
    # print("img_as_array.shape =", img_as_array.shape)
    # print("embedding.shape =", embedding.shape)
    embedding[height_start:height_end, width_start:width_end, :] = img_as_array
    return embedding


def predict_video(model_name, checkpoint_path, video_path, save_dir):

    idx_to_class_name = {0: 'NoViolence',
                         0.5: 'Uncertain',
                         1: 'Violence'}

    # class_name_to_idx = {v: k for k, v in idx_to_class_name.items()}

    lbl_name = video_path.split(os.sep)[-1].split('_')[0]
    lbl_cls_ID = 0 if lbl_name == "NONVIOLENCE" else 1


    model = Video_Classification_Models(model_name=model_name, num_classes=2)
    model = model.to(device)

    if os.path.isfile(checkpoint_path):
        print("=> Looking for checkpoint")
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        print("Loaded epoch number is", epoch)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> Found and loaded checkpoint")
    else:
        print("=> No checkpoint found. You have to train first.")
        sys.exit()


    frame_counter = 0
    WIN_SIZE = 16
    STEP_SIZE = 8
    IMG_SIZE = 256

    vid_cap = cv2.VideoCapture(video_path)
    suc, img = vid_cap.read()
    window = []
    all_frames = []
    # cv2.imshow("temp", img)
    # cv2.waitKey()
    preds_dict = {"classes" : [],
                  "conf": []}

    classes_per_win = []
    conf_per_win = []
    num_windows = 0
    torch.backends.cudnn.benchmark = True
    while suc:
        window.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        all_frames.append(img)
        if len(window) == WIN_SIZE:
            # inputs = torch.stack([cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) for frame in window])
            # input_numpy = np.array([cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) for frame in window], dtype=np.uint8)
            input_numpy = np.array(window, dtype=np.uint8)
            input_torch = torch.FloatTensor(input_numpy).permute(3, 0, 1, 2).unsqueeze(0)

            outputs = model(input_torch.to(device))
            # _, preds = torch.max(outputs, 1)

            y_pred_softmax = torch.nn.functional.softmax(outputs, dim=1)
            y_pred_softmax_numpy = y_pred_softmax[0].detach().cpu().numpy()
            # pred = output[0].permute(1, 2, 0).detach().cpu().numpy()
            _, pred_idx = torch.max(y_pred_softmax, 1)
            # pred_cls = class_map_idx_to_lbl[float(pred_idx)]
            pred_cls = int(pred_idx)
            conf = y_pred_softmax_numpy[pred_cls]
            preds_dict["classes"].extend([pred_cls] * WIN_SIZE)
            preds_dict["conf"].extend([conf] * WIN_SIZE)
            classes_per_win.append(pred_cls)
            conf_per_win.append(conf)

            num_windows += 1

            window = window[STEP_SIZE:]

        suc, img = vid_cap.read()

    # smoothed_confs = smoothen_transitions_conf_and_classes(preds_dict, window_size=WIN_SIZE, step_size=STEP_SIZE)
    preds_dict = my_smoothen_for_conf_and_cls(preds_dict, step_size=STEP_SIZE)
    write_video_with_pred(all_frames, preds_dict, save_dir, video_path, idx_to_class_name)

    """
    "lbl_name": lbl_name,
                  "lbl_cls_ID": lbl_cls_ID,
                  "num_windows": 0
    """

    preds_dict["lbl_name"] = lbl_name
    preds_dict["lbl_cls_ID"] = lbl_cls_ID
    preds_dict["num_windows"] = num_windows
    preds_dict["classes_per_win"] = classes_per_win
    preds_dict["conf_per_win"] = conf_per_win


    return preds_dict


def write_video_with_pred(all_frames, preds_dict, save_dir, video_path, idx_to_class_name):
    filename = '.'.join(video_path.split(os.sep)[-1].split('.')[:-1])
    height, width, _ = all_frames[0].shape


    max_side = np.max(all_frames[0].shape)
    ratio = 1280 / max_side
    new_height = int(height * ratio)
    new_width = int(width * ratio)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_dir + filename + ".mp4", fourcc, 30, (new_width, new_height))

    # for idx, frame in enumerate(all_frames):
    for idx in range(len(preds_dict["classes"])):
        frame = all_frames[idx]
        frame = cv2.resize(frame, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)

        temp = idx_to_class_name[preds_dict["classes"][idx]]
        img = draw_text_with_background(image=frame,
                                        class_name=temp,
                                        conf=preds_dict["conf"][idx],
                                        # text=idx_to_class_name[0],
                                        position=(25, 25),  # (width, height)
                                        font_scale=1.5)
        cv2.imshow("temp", img)
        cv2.waitKey(1)
        video_writer.write(img)

    video_writer.release()

def draw_text_with_background(image, class_name, conf, position, font_scale=1.0, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=3):
    # Definieren Sie die Schriftart und die Größe des Textes
    font_face = font
    # text_color = (0, 0, 250) if class_name == "Violence" else (0, 250, 0)

    if class_name == "Violence":
        text_color = (0, 0, 250)
    elif class_name == "Uncertain":
        text_color = (10, 200, 220)
    elif class_name == "NoViolence":
        text_color = (0, 250, 0)

    text = class_name + " " + str(round(conf * 100, 2)) + "%"

    # Die Größe des Texts berechnen, um die Größe des Hintergrundrechtecks zu bestimmen
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

    # Position des Textes festlegen (links oben)
    x, y = position

    # Hintergrundrechteck zeichnen
    background_color = (255, 255, 255)  # Weiß
    cv2.rectangle(image, (x, y), (x + text_width, y + text_height + baseline), background_color, -1)  # Hintergrund ausfüllen

    # Text zeichnen
    offset_vertikal = baseline // 2
    offset_horizontal = 1
    # text_color = (0, 0, 0)  # Schwarz
    cv2.putText(image, text, (x + offset_horizontal, y + text_height + offset_vertikal), font_face, font_scale, text_color, thickness)

    return image



def my_smoothen_for_conf_and_cls(preds_dict, step_size):
    conf_helper_lists = [[preds_dict["conf"][i * step_size + j] for j in range(step_size)] for i in range(len(preds_dict["conf"]) // step_size)]
    cls_helper_lists = [[preds_dict["classes"][i * step_size + j] for j in range(step_size)] for i in range(len(preds_dict["conf"]) // step_size)]

    final_dict = {"conf" : [*conf_helper_lists[0]],
                  "classes": [np.median(np.array(cls_helper_lists[0]))] * step_size}

    first_helper_list_idx = 1
    second_helper_list_idx = 2
    while second_helper_list_idx < len(conf_helper_lists) - 1:
        for i in range(step_size):
            a = conf_helper_lists[first_helper_list_idx][i]
            b = conf_helper_lists[second_helper_list_idx][i]
            temp = (a + b) / 2
            final_dict["conf"].append(temp)

        list_1 = cls_helper_lists[first_helper_list_idx]
        list_2 = cls_helper_lists[second_helper_list_idx]
        temp = np.median(np.array(list_1 + list_2))
        final_dict["classes"].extend([temp] * step_size)

        first_helper_list_idx += 2
        second_helper_list_idx += 2

    # Die letzten 8 Einträge bleiben gleich
    final_dict["conf"].extend(conf_helper_lists[-1])
    final_dict["classes"].extend([np.median(np.array(cls_helper_lists[-1]))] * step_size)

    return final_dict






def smoothen_transitions_conf_and_classes(preds_dict, window_size, step_size):
    preds_dict_new = {"classes": [],
                      "conf": []}
    num_windows = (len(preds_dict["conf"]) - window_size) // step_size + 1
    use_weights = True
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        conf_predictions = preds_dict["conf"][start_idx:end_idx]
        cls_predictions = preds_dict["classes"][start_idx:end_idx]

        conf_mean_preds = np.mean(conf_predictions, axis=0)
        cls_median_preds = np.median(cls_predictions, axis=0)

        preds_dict_new["conf"].append(np.mean(conf_mean_preds))
        preds_dict_new["classes"].append(np.median(cls_median_preds))

    return preds_dict_new




def smoothen_confidence_through_window_torch(predictions, kernel_size, stride):
    kernel_values = [1.0/kernel_size] * kernel_size
    kernel_tensor = torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Convolution durchführen
    conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
    conv.weight.data = kernel_tensor  # Kernel ändern
    predictions = predictions.unsqueeze(0)
    # output = conv(predictions)
    output = conv(predictions)

    return output

def smoothen_confidence_through_window_numpy(preds_dict, kernel_size, stride):
    kernel = np.ones(kernel_size) / kernel_size
    preds_dict["conf"] = np.convolve(preds_dict["conf"], kernel, mode='valid')[::stride]
    return preds_dict

