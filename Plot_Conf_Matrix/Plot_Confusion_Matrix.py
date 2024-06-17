import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

np.set_printoptions(precision=4, suppress=True)

def f1_for_multiclass_given_conf_matrix(conf_matrix):
    num_classes = len(conf_matrix)

    """
    we are computing macro F1, which is to compute the F1 of each class and then do the mean of those values
        -> micro is different. If I am not mistaken it would be using the avg prec and avg recall to calculate F1
    """

    Recall_list = []
    Precision_list = []
    F1_list = []
    for i in range(len(conf_matrix)):
        TP_cls_i = conf_matrix[i,i]
        FP_cls_i = sum(conf_matrix[i,:])-TP_cls_i
        FN_cls_i = sum(conf_matrix[:,i])-TP_cls_i
        Recall_i = TP_cls_i / (TP_cls_i + FN_cls_i)
        Precision_i = TP_cls_i / (TP_cls_i + FP_cls_i)
        F1_i = 2 * (Precision_i * Recall_i)/(Precision_i + Recall_i)

        if np.isnan(Recall_i):
            Recall_i = 0
        if np.isnan(Precision_i):
            Precision_i = 0
        if np.isnan(F1_i):
            F1_i = 0

        print("TP_cls_i =", TP_cls_i)
        print("FP_cls_i =", FP_cls_i)
        print("FN_cls_i =", FN_cls_i)
        print("Recall_i =", Recall_i)
        print("Precision_i =", Precision_i)
        print("F1_i =", F1_i)

        Recall_list.append(Recall_i)
        Precision_list.append(Precision_i)
        F1_list.append(F1_i)

    average_Recall = np.array(Recall_list).mean()
    average_Precision = np.array(Precision_list).mean()
    average_F1 = np.array(F1_list).mean()

    use_round = False
    if use_round:
        return (round(average_Recall, 2), round(average_Precision, 2), round(average_F1, 2))
    else:
        return (average_Precision, average_Recall, average_F1)



def plot_confusion_matrix(conf_matrix, save_path, idx_to_class_name, normalize=True, missing_classes=[], plot_colorbar=True, plot_conf_matrix=True):
    # fig = plt.figure(figsize=(16,9))
    # missing_classes = list(set([2, 1, 3, 3]))
    missing_classes = sorted(missing_classes)
    print("delete_indices =", missing_classes)
    reversed_missing_classes = missing_classes[::-1]
    print("reversed_delete_idx =", reversed_missing_classes)
    print("conf_matrix =", conf_matrix)

    for i in reversed_missing_classes:
        conf_matrix = np.delete(conf_matrix, i, 0)
        conf_matrix = np.delete(conf_matrix, i, 1)

    print("conf_matrix *** Before Normalizing *** =", conf_matrix)
    # cax = ax.matshow(conf_matrix, cmap=mpl.get_cmap('viridis', 256))

    normalize_values = []
    for i in range(len(conf_matrix)):
        sum_col_i = sum(conf_matrix[:, i])
        normalize_values.append(sum_col_i)
    print("normalize_values =", normalize_values)

    conf_matrix_normalized = conf_matrix.copy()
    for i in range(len(idx_to_class_name)):
        for j in range(len(idx_to_class_name)):
            if normalize_values[j] != 0:
                c_norm = round(100.0 * float(conf_matrix[i, j]) / float(normalize_values[j]), 2)
                conf_matrix_normalized[i,j] = c_norm
            else:
                print("Attention, at least one class does not contain any GT data.")

    print("conf_matrix_normalized *** After Normalizing *** =", conf_matrix_normalized)

    metrics = f1_for_multiclass_given_conf_matrix(conf_matrix)
    print("save_path =", save_path)
    print("metrics =", metrics)


    fig = plt.figure("Heatmap - Confusionmatrix")
    scaling_factor = 1.25
    # fig.set_size_inches(w=8*scaling_factor, h=6*scaling_factor)
    fig.set_size_inches(w=8*scaling_factor, h=6*scaling_factor)
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix_normalized)

    # Festlegen der benutzerdefinierten Ticks für die Farbskala der Colorbar
    if plot_colorbar:
        # cbar = plt.colorbar(cax, ticks=np.linspace(0, 100, 11))
        cbar = plt.colorbar(cax, ticks=np.linspace(0, 100, 11))
        cbar.set_label('Normalized Values (%)')


    # maxi = np.amax(conf_matrix)
    # mini = np.amin(conf_matrix)

    fontsize_percent_values = 14
    fontsize_samples_values = 14
    fontsize_labels = 16
    fontsize_GT_text = 16
    fontsize_Pred_text = 16
    fontsize_header = 18



    # labels = [classes_dict[elem] for elem in range(num_classes)]
    labels = [idx_to_class_name[cls_id] for cls_id in sorted(idx_to_class_name.keys())]
    org_dif = -0.125
    total_dif = 0.125
    max_normalized_value = np.max(conf_matrix_normalized)
    for i in range(len(labels)):
        for j in range(len(labels)):
            # c_norm = round(100.0 * float(conf_matrix[j, i]) / float(normalize_values[i]), 2)
            c = round(conf_matrix_normalized[i, j], 2)
            percentage_str = str(round(conf_matrix_normalized[i, j], 2)) + "%"
            total_num = "(" + str(int(conf_matrix[i,j])) + ")"
            if c > max_normalized_value * 0.6:
                ax.text(j, i+org_dif, percentage_str, va='center', ha='center', color=(0, 0, 0), fontsize=fontsize_percent_values)
                ax.text(j, i+total_dif, total_num, va='center', ha='center', color=(0, 0, 0), fontsize=fontsize_samples_values)
            else:
                ax.text(j, i+org_dif, percentage_str, va='center', ha='center', color=(0.99, 0.9, 0.14), fontsize=fontsize_percent_values)
                ax.text(j, i+total_dif, total_num, va='center', ha='center', color=(0.99, 0.9, 0.14), fontsize=fontsize_samples_values)

    # title = "Confusion Matrix"
    # plt.title(title)
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize_labels)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=fontsize_labels) # center, right
    ax.set_yticklabels(labels, rotation=0, ha='right', fontsize=fontsize_labels)

    # Manuelle Anpassung der Position der Achsenticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # ax.set_yticklabels([''] + labels)

    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    # header = r"$Macro\; F_1-Score = \it{}$".format(str(metrics[2]))
    header = "$Macro\ F_1-Score = " + str(round(metrics[2]*100, 2)) + "$%"
    # +r"$\bf{" + var + "}$"
    # plt.title(header, loc=(0.0, 0.5), va='center', ha='center', color=(0, 0, 0))
    # fig.suptitle(header, fontsize=12, fontweight='bold')
    print("*********************************")
    print(header)
    print("*********************************")

    # fig.suptitle(header, fontsize=13)


    # plt.xlabel("\nGround Truth")
    # plt.ylabel("\nPredicted")

    num_classes = len(labels)

    # Statt hart codierten Werten die Positionen automatisch berechnen
    ground_truth_x = 0.5 # num_classes // 2
    ground_truth_y = -0.6
    predicted_x = -1.10 # -3.5
    predicted_y = 0.5 # num_classes // 2

    # plt.text(6, -1, "Ground Truth", va='center', ha='center')
    # plt.text(-6, 6, "Predicted", rotation=90, va='center', ha='center')
    plt.text(ground_truth_x, ground_truth_y, "Ground Truth", va='center', ha='center', fontsize=fontsize_GT_text)
    plt.text(ground_truth_x, ground_truth_y - 0.25, header, va='center', ha='center', fontsize=fontsize_header)
    plt.text(predicted_x, predicted_y, "Predicted", rotation=90, va='center', ha='center', fontsize=fontsize_Pred_text)

    # plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad = 0.1, w_pad = 0.2, rect=(left=0.0, bottom=0.22, right=0.863, top=0.737))
    # plt.tight_layout(pad=0, h_pad = 0.1, w_pad = 0.2, left=0.0, bottom=0.22, right=0.863, top=0.737)
    # plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.subplots_adjust(left=0.09, right=0.995, bottom=0.06, top=0.835)

    plt.savefig(save_path + ".pdf")
    plt.savefig(save_path + ".png")
    plt.show()
