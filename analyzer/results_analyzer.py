import os
import numpy as np
import csv
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# metric score should be the string name of the coloumb in the metric_scores
# options are: micro_f1, main_role_accuracy, emr, micro_precision, micro_recall

def get_average_general_new(metric_score ,file_folder, threshold_list, lr_list, batchsize_list, epoch_list):

    averages = {}
    max = 0
    max_keys = {}


    for index, threshold in enumerate(threshold_list):
        averages[threshold] = {}
        for lr in lr_list:
            averages[threshold][lr] = {}

            for epochs in epoch_list:
                averages[threshold][lr][epochs] = {}
                for batchsize in batchsize_list:

                    list = []
                    for iteration in range(0, 10):
                        csvFile = pd.read_csv(file_folder + "/metric_scores_lr_" + str(lr) +"_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) + ".csv")

                        list.append(csvFile[metric_score][index])

                    averages[threshold][lr][epochs][batchsize] = np.mean(list)


                    if averages[threshold][lr][epochs][batchsize] > max:
                        max = averages[threshold][lr][epochs][batchsize]
                        max_keys = {
                            "lr": lr,
                            "threshold": threshold,
                            "batchsize": batchsize,
                            "epochs":epochs

                        }

    with open(file_folder + "/" + metric_score + "_averages.json", "w") as file:
        json.dump(averages, file, indent=4)


    return averages, max_keys, max


def heat_map_lr_batchsize(metric_average_dict, threshold_list, lr_list, batchsize_list, epochs_list, title, dir_name):
    try:
        os.mkdir(dir_name)
    except:
        print("dir already exists!")
    for threshold in threshold_list:
        data = np.zeros((len(lr_list), len(batchsize_list)))
        for index_lr, lr in enumerate(lr_list):
            for index_epoch, epochs in enumerate(epochs_list):
                for index_batch, batchsize in enumerate(batchsize_list):
                    data[index_lr][index_batch] = metric_average_dict[threshold][lr][epochs][batchsize]


        plt.figure(figsize=(8, 6))  # Optionale Größe des Plots
        sns.heatmap(data, annot=True, cmap="YlOrBr", fmt=".3f", xticklabels=batchsize_list, yticklabels=lr_list)  # annot=True für Werte anzeigen
        plt.title(title + " with threshold: " + str(threshold))
        plt.xlabel("batch size")
        plt.ylabel("learning rate")
        plt.savefig(dir_name + "/" + title + " with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()

def heat_map_lr_batchsize_new(metrics, metrics_dict,metrics_dir, threshold_list, lr_list, batchsize_list, epochs_list, title, dir_name):
    try:
        os.mkdir(dir_name)
    except:
        print("dir already exists!")

    if metrics == None:
        with open(metrics_dir + ".json", "r") as file:
            metrics_dict = json.load(file)

    for threshold in threshold_list:
        data = np.zeros((len(lr_list), len(batchsize_list)))
        for index_lr, lr in enumerate(lr_list):
            for index_epoch, epochs in enumerate(epochs_list):
                for index_batch, batchsize in enumerate(batchsize_list):
                    data[index_lr][index_batch] = metrics_dict[threshold][lr][epochs][batchsize][metrics]


        plt.figure(figsize=(8, 6))  # Optionale Größe des Plots
        sns.heatmap(data, annot=True, cmap="YlOrBr", fmt=".3f", xticklabels=batchsize_list, yticklabels=lr_list)  # annot=True für Werte anzeigen
        plt.title(title + " with threshold: " + str(threshold))
        plt.xlabel("batch size")
        plt.ylabel("learning rate")
        plt.savefig(dir_name + "/" + title + " with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()

def heat_map_lr_epochs_new(metric_average_dict, threshold_list, lr_list, batchsize_list, epochs_list, title, dir_name):
    try:
        os.mkdir(dir_name)
    except:
        print("dir already exists!")
    for threshold in threshold_list:
        data = np.zeros((len(lr_list), len(epochs_list)))
        for index_lr, lr in enumerate(lr_list):
            for index_epoch, epochs in enumerate(epochs_list):
                for index_batch, batchsize in enumerate(batchsize_list):
                    print("index_lr: ",index_lr)
                    print("index_epoch: ", index_epoch)
                    print(metric_average_dict[threshold][lr][epochs][batchsize])
                    data[index_lr][index_epoch] = metric_average_dict[threshold][lr][epochs][batchsize]


        plt.figure(figsize=(8, 6))  # Optionale Größe des Plots
        sns.heatmap(data, annot=True, cmap="YlOrBr", fmt=".3f", xticklabels=epochs_list, yticklabels=lr_list)  # annot=True für Werte anzeigen
        plt.title(title + " with threshold: " + str(threshold))
        plt.xlabel("epochs")
        plt.ylabel("learning rate")
        plt.savefig(dir_name + "/" + title + " with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()


#----------------max_roles:


def get_most_frequent_roles_new(file_folder, threshold_list, lr_list, batchsize_list, epoch_list):

    role_dict = {}
    num_rows = 1
    for index, threshold in enumerate(threshold_list):
        role_dict[threshold] = {}
        for lr in lr_list:
            role_dict[threshold][lr] = {}
            for epochs in epoch_list:
                role_dict[threshold][lr][epochs] = {}
                for batchsize in batchsize_list:
                    role_dict[threshold][lr][epochs][batchsize] = {}
                    role_dict[threshold][lr][epochs][batchsize]["main_role"] = {}
                    role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"] = {}

                    for iteration in range(0, 10):

                        print("csvFile: ",
                              file_folder + "/" + file_folder + "_threshold_" + str(threshold) + "_lr_" + str(
                                  lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(
                                  iteration) + ".tsv")
                        csvFile = pd.read_csv(file_folder + "/" + file_folder + "_threshold_" + str(threshold) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) + ".tsv", sep="\t")


                        num_rows = csvFile.shape[0]

                        for index, line in csvFile.iterrows():
                            #print("index: ",index)
                            #print("line: ", line)

                            if csvFile["main_role"][index] in role_dict[threshold][lr][epochs][batchsize]["main_role"].keys() :
                                role_dict[threshold][lr][epochs][batchsize]["main_role"][csvFile["main_role"][index]] =  role_dict[threshold][lr][epochs][batchsize]["main_role"][csvFile["main_role"][index]] +1
                            else:
                                role_dict[threshold][lr][epochs][batchsize]["main_role"][csvFile["main_role"][index]] = 1

                            if csvFile["fine_grained_roles_1"][index] in role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"].keys() :
                                role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][csvFile["fine_grained_roles_1"][index]] =  role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][csvFile["fine_grained_roles_1"][index]] +1
                            else:
                                role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][csvFile["fine_grained_roles_1"][index]] = 1

                            if csvFile["fine_grained_roles_2"][index] in role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"].keys() and not csvFile["fine_grained_roles_2"][index] == "NaN":
                                role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][csvFile["fine_grained_roles_2"][index]] =  role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][csvFile["fine_grained_roles_2"][index]] +1
                            else:
                                role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][csvFile["fine_grained_roles_2"][index]] = 1


                    max_value = 0
                    max_class = ""
                    for key in (role_dict[threshold][lr][epochs][batchsize]["main_role"]).keys():

                        role_dict[threshold][lr][epochs][batchsize]["main_role"][key] = role_dict[threshold][lr][epochs][batchsize]["main_role"][key] / 10
                        if role_dict[threshold][lr][epochs][batchsize]["main_role"][key] > max_value:
                            max_value = role_dict[threshold][lr][epochs][batchsize]["main_role"][key]
                            max_class = key

                    role_dict[threshold][lr][epochs][batchsize]["main_role"]["max_value"] = round((max_value / num_rows) * 100, 2) # value in percentage of all test-data
                    role_dict[threshold][lr][epochs][batchsize]["main_role"]["max_class"] = max_class


                    max_value_sub_role_1 = 0
                    max_value_sub_role_2 = 0
                    max_class_sub_role_1 = ""
                    max_class_sub_role_2 = ""

                    for key in (role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]).keys():

                        role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][key] = role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][key] / 10
                        if role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][key] > max_value_sub_role_1:
                            max_value_sub_role_1 = role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][key]
                            max_class_sub_role_1 = key
                        elif role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][key] > max_value_sub_role_2:
                            max_value_sub_role_2 = role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][key]
                            max_class_sub_role_2 = key


                    role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_1"] = round((max_value_sub_role_1 / num_rows) * 100, 2) # value in percentage of all test-data
                    role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_class_sub_role_1"] = max_class_sub_role_1
                    role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_2"] = round((max_value_sub_role_2 / num_rows) * 100, 2)  # value in percentage of all test-data
                    role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_class_sub_role_2"] = max_class_sub_role_2

    return role_dict


def heat_map_max_role_frequency_lr_batchsize_new(role_dict, threshold_list, lr_list, batchsize_list,epochs, title, dir_name):
    #print(role_dict)
    try:
        os.mkdir(dir_name)
    except:
        print("dir already exists!")

    for threshold in threshold_list:
        data_main_role = []
        data_fine_grained_role_1 = []
        data_fine_grained_role_2 = []
        dummy_data_main_role = np.zeros((len(lr_list), len(batchsize_list)))
        dummy_data_fine_grained_role_1 = np.zeros((len(lr_list), len(batchsize_list)))
        dummy_data_fine_grained_role_2 = np.zeros((len(lr_list), len(batchsize_list)))
        for index_lr, lr in enumerate(lr_list):
            temp_list_main_role = []
            temp_list_fine_grained_role_1 = []
            temp_list_fine_grained_role_2 = []


            for index_batch, batchsize in enumerate(batchsize_list):
                dummy_data_main_role[index_lr][index_batch] = role_dict[threshold][lr][epochs][batchsize]["main_role"]["max_value"]
                temp_list_main_role.append(str(role_dict[threshold][lr][epochs][batchsize]["main_role"]["max_class"]) + " : " + str(role_dict[threshold][lr][epochs][batchsize]["main_role"]["max_value"]) + "%")

                dummy_data_fine_grained_role_1[index_lr][index_batch] = role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_1"]
                temp_list_fine_grained_role_1.append(
                    str(role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_class_sub_role_1"]) + " : " + str(
                        role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_1"]) + "%")

                dummy_data_fine_grained_role_2[index_lr][index_batch] = role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_2"]
                temp_list_fine_grained_role_2.append(
                    str(role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][
                            "max_class_sub_role_2"]) + " : " + str(
                        role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_2"]) + "%")


            data_main_role.append(temp_list_main_role)
            data_fine_grained_role_1.append(temp_list_fine_grained_role_1)
            data_fine_grained_role_2.append(temp_list_fine_grained_role_2)



        plt.figure(figsize=(14, 12))  # Optionale Größe des Plots
        sns.heatmap(dummy_data_main_role, annot=data_main_role, cmap="YlOrBr",fmt="", xticklabels=batchsize_list, yticklabels=lr_list, vmin=0, vmax=100)  # annot=True für Werte anzeigen
        plt.title(title + " main-role with threshold: " + str(threshold))
        plt.xlabel("batch size")
        plt.ylabel("learning rate")

        # Heatmap als PNG speichern
        plt.savefig(dir_name + "/" + title + " main-role with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()

        plt.figure(figsize=(14, 12))  # Optionale Größe des Plots
        sns.heatmap(dummy_data_fine_grained_role_1, annot=data_fine_grained_role_1, cmap="YlOrBr", fmt="", xticklabels=batchsize_list,
                    yticklabels=lr_list, vmin=0, vmax=100)  # annot=True für Werte anzeigen
        plt.title(title + " sub-role 1 with threshold: " + str(threshold))
        plt.xlabel("batch size")
        plt.ylabel("learning rate")

        plt.savefig(dir_name + "/" + title + " sub-role 1 with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()

        plt.figure(figsize=(14, 12)) # Optionale Größe des Plots
        sns.heatmap(dummy_data_fine_grained_role_2, annot=data_fine_grained_role_2, cmap="YlOrBr", fmt="", xticklabels=batchsize_list,
                    yticklabels=lr_list, vmin=0, vmax=100)  # annot=True für Werte anzeigen
        plt.title(title + " sub-role 2 with threshold: " + str(threshold))
        plt.xlabel("batch size")
        plt.ylabel("learning rate")
        plt.savefig(dir_name + "/" + title + " sub-role 2 with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()

def heat_map_max_role_frequency_lr_epochs_new(role_dict, threshold_list, lr_list, batchsize,epoch_list, title, dir_name):
    #print(role_dict)
    try:
        os.mkdir(dir_name)
    except:
        print("dir already exists!")

    for threshold in threshold_list:
        data_main_role = []
        data_fine_grained_role_1 = []
        data_fine_grained_role_2 = []
        dummy_data_main_role = np.zeros((len(lr_list), len(epoch_list)))
        dummy_data_fine_grained_role_1 = np.zeros((len(lr_list), len(epoch_list)))
        dummy_data_fine_grained_role_2 = np.zeros((len(lr_list), len(epoch_list)))
        for index_lr, lr in enumerate(lr_list):
            temp_list_main_role = []
            temp_list_fine_grained_role_1 = []
            temp_list_fine_grained_role_2 = []


            for index_epochs, epochs in enumerate(epoch_list):
                dummy_data_main_role[index_lr][index_epochs] = role_dict[threshold][lr][epochs][batchsize]["main_role"]["max_value"]
                temp_list_main_role.append(str(role_dict[threshold][lr][epochs][batchsize]["main_role"]["max_class"]) + " : " + str(role_dict[threshold][lr][epochs][batchsize]["main_role"]["max_value"]) + "%")

                dummy_data_fine_grained_role_1[index_lr][index_epochs] = role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_1"]
                temp_list_fine_grained_role_1.append(
                    str(role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_class_sub_role_1"]) + " : " + str(
                        role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_1"]) + "%")

                dummy_data_fine_grained_role_2[index_lr][index_epochs] = role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_2"]
                temp_list_fine_grained_role_2.append(
                    str(role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"][
                            "max_class_sub_role_2"]) + " : " + str(
                        role_dict[threshold][lr][epochs][batchsize]["fine_grained_roles"]["max_value_sub_role_2"]) + "%")


            data_main_role.append(temp_list_main_role)
            data_fine_grained_role_1.append(temp_list_fine_grained_role_1)
            data_fine_grained_role_2.append(temp_list_fine_grained_role_2)



        plt.figure(figsize=(14, 12))  # Optionale Größe des Plots
        sns.heatmap(dummy_data_main_role, annot=data_main_role, cmap="YlOrBr",fmt="", xticklabels=epoch_list, yticklabels=lr_list, vmin=0, vmax=100)  # annot=True für Werte anzeigen
        plt.title(title + " main-role with threshold: " + str(threshold))
        plt.xlabel("epochs")
        plt.ylabel("learning rate")

        # Heatmap als PNG speichern
        plt.savefig(dir_name + "/" + title + " main-role with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()

        plt.figure(figsize=(14, 12))  # Optionale Größe des Plots
        sns.heatmap(dummy_data_fine_grained_role_1, annot=data_fine_grained_role_1, cmap="YlOrBr", fmt="", xticklabels=epoch_list,
                    yticklabels=lr_list, vmin=0, vmax=100)  # annot=True für Werte anzeigen
        plt.title(title + " sub-role 1 with threshold: " + str(threshold))
        plt.xlabel("epochs")
        plt.ylabel("learning rate")

        plt.savefig(dir_name + "/" + title + " sub-role 1 with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()

        plt.figure(figsize=(14, 12)) # Optionale Größe des Plots
        sns.heatmap(dummy_data_fine_grained_role_2, annot=data_fine_grained_role_2, cmap="YlOrBr", fmt="", xticklabels=epoch_list,
                    yticklabels=lr_list, vmin=0, vmax=100)  # annot=True für Werte anzeigen
        plt.title(title + " sub-role 2 with threshold: " + str(threshold))
        plt.xlabel("epochs")
        plt.ylabel("learning rate")
        plt.savefig(dir_name + "/" + title + " sub-role 2 with threshold " + str(threshold) + ".png", format="png", dpi=300)
        plt.show()

def get_f1_per_main_role_new(file_folder, threshold_list, lr_list, batchsize_list, epoch_list):

    averages = {}


    for index, threshold in enumerate(threshold_list):
        averages[threshold] = {}
        for lr in lr_list:
            averages[threshold][lr] = {}

            for epochs in epoch_list:
                averages[threshold][lr][epochs] = {}
                for batchsize in batchsize_list:
                    averages[threshold][lr][epochs][batchsize] = {}
                    dict = {
                        "Antagonist":{"TP":0, "TN":0, "FP":0, "FN":0},
                        "Protagonist": {"TP":0, "TN":0, "FP":0, "FN":0},
                        "Innocent": {"TP":0, "TN":0, "FP":0, "FN":0}

                    }
                    for iteration in range(0, 10):
                        predicitions = pd.read_csv(file_folder + "/" + file_folder + "_threshold_" + str(threshold) + "_lr_" + str(lr) +"_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) + ".tsv", sep="\t")
                        gold_files = pd.read_csv(file_folder + "/gold_labels_lr_" + str(lr) + "_epochs_" + str(
                            epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) + ".tsv", sep="\t")

                        # Protagonist:
                        for index, line in predicitions.iterrows():
                            if line["main_role"] == gold_files["main_role"][index] and line["main_role"] == "Protagonist":
                                dict["Protagonist"]["TP"] += 1

                            elif line["main_role"] != gold_files["main_role"][index] and line["main_role"] == "Protagonist":
                                dict["Protagonist"]["FP"] += 1

                            elif gold_files["main_role"][index] == "Protagonist" and line["main_role"] != "Protagonist":
                                dict["Protagonist"]["FN"] += 1

                            else:
                                dict["Protagonist"]["TN"] += 1

                        # Antagonist:
                        for index, line in predicitions.iterrows():
                            if line["main_role"] == gold_files["main_role"][index] and line["main_role"] == "Antagonist":
                                dict["Antagonist"]["TP"] += 1

                            elif line["main_role"] != gold_files["main_role"][index] and line["main_role"] == "Antagonist":
                                dict["Antagonist"]["FP"] += 1

                            elif gold_files["main_role"][index] == "Antagonist" and line["main_role"] != "Antagonist":
                                dict["Antagonist"]["FN"] += 1

                            else:
                                dict["Antagonist"]["TN"] += 1

                        # Innocent:
                        for index, line in predicitions.iterrows():
                            if line["main_role"] == gold_files["main_role"][index] and line["main_role"] == "Innocent":
                                dict["Innocent"]["TP"] += 1

                            elif line["main_role"] != gold_files["main_role"][index] and line["main_role"] == "Innocent":
                                dict["Innocent"]["FP"] += 1

                            elif gold_files["main_role"][index] == "Innocent" and line["main_role"] != "Innocent":
                                dict["Innocent"]["FN"] += 1

                            else:
                                dict["Innocent"]["TN"] += 1


                    averages[threshold][lr][epochs][batchsize] = dict

                    if dict["Antagonist"]["TP"] != 0:
                        averages[threshold][lr][epochs][batchsize]["Antagonist"]["precision"] = dict["Antagonist"]["TP"] / (dict["Antagonist"]["TP"] + dict["Antagonist"]["FP"])
                        averages[threshold][lr][epochs][batchsize]["Antagonist"]["recall"] = dict["Antagonist"]["TP"] / (dict["Antagonist"]["TP"] + dict["Antagonist"]["FN"])
                        averages[threshold][lr][epochs][batchsize]["Antagonist"]["f1"] = (2 * dict["Antagonist"]["precision"] * dict["Antagonist"]["recall"]) / (dict["Antagonist"]["precision"] + dict["Antagonist"]["recall"])
                    else:
                        averages[threshold][lr][epochs][batchsize]["Antagonist"]["precision"] = 0
                        averages[threshold][lr][epochs][batchsize]["Antagonist"]["recall"] = 0
                        averages[threshold][lr][epochs][batchsize]["Antagonist"]["f1"] = 0



                    if dict["Protagonist"]["TP"] != 0:
                        averages[threshold][lr][epochs][batchsize]["Protagonist"]["precision"] = dict["Protagonist"]["TP"] / (dict["Protagonist"]["TP"] + dict["Protagonist"]["FP"])
                        averages[threshold][lr][epochs][batchsize]["Protagonist"]["recall"] = dict["Protagonist"]["TP"] / (dict["Protagonist"]["TP"] + dict["Protagonist"]["FN"])
                        averages[threshold][lr][epochs][batchsize]["Protagonist"]["f1"] = (2 * dict["Protagonist"]["precision"] * dict["Protagonist"]["recall"]) / (dict["Protagonist"]["precision"] +dict["Protagonist"]["recall"])
                    else:
                        averages[threshold][lr][epochs][batchsize]["Protagonist"]["precision"] = 0
                        averages[threshold][lr][epochs][batchsize]["Protagonist"]["recall"] = 0
                        averages[threshold][lr][epochs][batchsize]["Protagonist"]["f1"] = 0

                    if dict["Innocent"]["TP"] != 0:
                        averages[threshold][lr][epochs][batchsize]["Innocent"]["precision"] = dict["Innocent"]["TP"] / (dict["Innocent"]["TP"] + dict["Innocent"]["FP"])
                        averages[threshold][lr][epochs][batchsize]["Innocent"]["recall"] = dict["Innocent"]["TP"] / (dict["Innocent"]["TP"] + dict["Innocent"]["FN"])
                        averages[threshold][lr][epochs][batchsize]["Innocent"]["f1"] = (2 * dict["Innocent"]["precision"] * dict["Innocent"]["recall"]) / (dict["Innocent"]["precision"] +dict["Innocent"]["recall"])
                    else:
                        averages[threshold][lr][epochs][batchsize]["Innocent"]["precision"] = 0
                        averages[threshold][lr][epochs][batchsize]["Innocent"]["recall"] = 0
                        averages[threshold][lr][epochs][batchsize]["Innocent"]["f1"] = 0


                    TP = dict["Innocent"]["TP"] + dict["Protagonist"]["TP"] + dict["Antagonist"]["TP"]
                    FP = dict["Innocent"]["FP"] + dict["Protagonist"]["FP"] + dict["Antagonist"]["FP"]
                    FN = dict["Innocent"]["FN"] + dict["Protagonist"]["FN"] + dict["Antagonist"]["FN"]
                    TN = dict["Innocent"]["TN"] + dict["Protagonist"]["TN"] + dict["Antagonist"]["TN"]

                    mp = TP/(TP+FP)
                    mr = TP/(TP+FN)

                    averages[threshold][lr][epochs][batchsize]["micro_precision"] = mp
                    averages[threshold][lr][epochs][batchsize]["micro_recall"] = mr
                    averages[threshold][lr][epochs][batchsize]["micro_f1"] = (2 * mr * mp) / (mr + mp)
                    averages[threshold][lr][epochs][batchsize]["accuracy"] = (TP + TN) / (TP + TN + FP + FN)




    with open(file_folder + "/metrics_per_class_averages.json", "w") as file:
        json.dump(averages, file, indent=4)


    return averages

def get_emr_per_role_new(file_folder, threshold_list, lr_list, batchsize_list, epoch_list):

    averages = {}


    for index, threshold in enumerate(threshold_list):
        averages[threshold] = {}
        for lr in lr_list:
            averages[threshold][lr] = {}

            for epochs in epoch_list:
                averages[threshold][lr][epochs] = {}
                for batchsize in batchsize_list:
                    averages[threshold][lr][epochs][batchsize] = {}
                    exact_matches = 0
                    exact_matches_partial = 0
                    num_instances = 0
                    num_instances_partial = 0
                    for iteration in range(0, 10):
                        predicitions = pd.read_csv(file_folder + "/" + file_folder + "_threshold_" + str(threshold) + "_lr_" + str(lr) +"_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) + ".tsv", sep="\t")
                        gold_files = pd.read_csv(file_folder + "/gold_labels_lr_" + str(lr) + "_epochs_" + str(
                            epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) + ".tsv", sep="\t")



                        # Protagonist:
                        for index, line in predicitions.iterrows():
                            num_instances += 1
                            prediciton_list = [line["fine_grained_roles_1"], line["fine_grained_roles_2"]]
                            gold_list = [gold_files["fine_grained_roles_1"][index], gold_files["fine_grained_roles_2"][index]]
                            if set(prediciton_list) == set(gold_list):
                                exact_matches += 1
                                exact_matches_partial += 1
                            elif prediciton_list[0] in gold_list or prediciton_list[1] in gold_list:
                                exact_matches_partial += 0.5

                            if gold_list[1] != "NaN":
                                num_instances_partial += 1
                            else:
                                num_instances_partial += 0.5


                    averages[threshold][lr][epochs][batchsize]["emr"] = exact_matches / num_instances
                    averages[threshold][lr][epochs][batchsize]["emr_partial"] = exact_matches_partial / num_instances_partial



    with open(file_folder + "/my_emr_per_sub_class.json", "w") as file:
        json.dump(averages, file, indent=4)


    return averages

if __name__ == '__main__':
    threshold_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.5, 0.7]#[0.0005, 0.001, 0.005, 0.01, 0.05]#[0.05, 0.075, 0.1]
    lr_list = [3e-5, 4e-5, 5e-5, 6e-5]#[8e-5, 1e-4, 1e-3, 1e-2]#[4e-5, 5e-5, 6e-5]  # [1e-5, 2e-5, 3e-5]
    batchsize_list = [8]#[8]  # [8, 16, 32]
    epoch_list = [6, 10, 15, 20]#[6, 7, 8]#[5, 7, 10]
    #epochs = 3
    batchsize = 8
    #emr,micro_precision,micro_recall,micro_f1,main_role_accuracy
    file_folder = "roberta_epochs_lr_threshold"
    dir_name = "heatmaps_roberta_epochs_lr_threshold"




    micro_f1_averages, max_micro_f1_keys, max_micro_f1 = get_average_general_new(
        "micro_f1",
        file_folder, threshold_list, lr_list, batchsize_list, epoch_list)


    heat_map_lr_epochs_new(micro_f1_averages, threshold_list, lr_list, batchsize_list, epoch_list,
                              "micro-f1 averages", dir_name)

    main_role_accuracy_averages, max_main_role_accuracy_keys, max_main_role_accuracy = get_average_general_new("emr",
                                                                                                               file_folder,
                                                                                                               threshold_list,
                                                                                                               lr_list,
                                                                                                               batchsize_list,
                                                                                                               epoch_list)

    heat_map_lr_epochs_new(main_role_accuracy_averages, threshold_list, lr_list, batchsize_list, epoch_list,
                          "exact match ratio", dir_name)

    main_role_accuracy_averages, max_main_role_accuracy_keys, max_main_role_accuracy = get_average_general_new(
        "main_role_accuracy",
        file_folder, threshold_list, lr_list, batchsize_list, epoch_list)

    heat_map_lr_epochs_new(main_role_accuracy_averages, threshold_list, lr_list, batchsize_list, epoch_list,
                           "accuracy", dir_name)


    main_role_accuracy_averages, max_main_role_accuracy_keys, max_main_role_accuracy = get_average_general_new(
        "micro_precision",
        file_folder, threshold_list, lr_list, batchsize_list, epoch_list)


    heat_map_lr_epochs_new(main_role_accuracy_averages, threshold_list, lr_list, batchsize_list, epoch_list,
                          "micro-precision", dir_name)



    main_role_accuracy_averages, max_main_role_accuracy_keys, max_main_role_accuracy = get_average_general_new(
        "micro_recall",
        file_folder, threshold_list, lr_list, batchsize_list, epoch_list)


    heat_map_lr_epochs_new(main_role_accuracy_averages, threshold_list, lr_list, batchsize_list, epoch_list,
                          "micro-recall", dir_name)

    dict = get_most_frequent_roles_new(file_folder, threshold_list, lr_list, batchsize_list, epoch_list)
    heat_map_max_role_frequency_lr_epochs_new(dict, threshold_list, lr_list, batchsize, epoch_list,
                                                 "most frequent classes", dir_name)

    '''micro_f1_averages, max_micro_f1_keys, max_micro_f1 = get_average_general_new(
        "micro_f1",
        file_folder, threshold_list, lr_list, batchsize_list, epoch_list)

    print("micro_f1_averages: ", micro_f1_averages)
    print("max_micro_f1_keys: ",max_micro_f1_keys)
    print("max_micro_f1: ", max_micro_f1)
    heat_map_lr_batchsize(micro_f1_averages, threshold_list, lr_list, batchsize_list, epoch_list,
                              "micro-f1 averages", dir_name)
    heat_map_lr_batchsize_new("micro_f1",averages,None, threshold_list, lr_list, batchsize_list,epoch_list, "main-role micro-f1 averages", dir_name)



    main_role_accuracy_averages, max_main_role_accuracy_keys, max_main_role_accuracy = get_average_general_new("emr",
                                                                                                           file_folder, threshold_list, lr_list, batchsize_list, epoch_list)
    print("emr_averages: ", main_role_accuracy_averages)
    print("max_emr_keys: ", max_main_role_accuracy_keys)
    print("max_emr: ", max_main_role_accuracy)
    heat_map_lr_batchsize(main_role_accuracy_averages, threshold_list, lr_list, batchsize_list, epoch_list,"exact match ratio",dir_name)

    main_role_accuracy_averages, max_main_role_accuracy_keys, max_main_role_accuracy = get_average_general_new(
        "main_role_accuracy",
        file_folder, threshold_list, lr_list, batchsize_list, epoch_list)
    print("main_role_accuracy_averages: ", main_role_accuracy_averages)
    print("max_main_role_accuracy_keys: ", max_main_role_accuracy_keys)
    print("max_main_role_accuracy: ", max_main_role_accuracy)
    heat_map_lr_batchsize(main_role_accuracy_averages, threshold_list, lr_list, batchsize_list,epoch_list, "main-role accuracy",dir_name)

    main_role_accuracy_averages, max_main_role_accuracy_keys, max_main_role_accuracy = get_average_general_new(
        "micro_precision",
        file_folder, threshold_list, lr_list, batchsize_list, epoch_list)
    print("micro_precision_averages: ", main_role_accuracy_averages)
    print("max_micro_precision_keys: ", max_main_role_accuracy_keys)
    print("max_micro_precision: ", max_main_role_accuracy)

    heat_map_lr_batchsize(main_role_accuracy_averages, threshold_list, lr_list, batchsize_list, epoch_list,"micro-precision", dir_name)

    heat_map_lr_batchsize_new("micro_precision",averages,None, threshold_list, lr_list, batchsize_list,epoch_list, "main-roles micro-precision",
                          dir_name)

    main_role_accuracy_averages, max_main_role_accuracy_keys, max_main_role_accuracy = get_average_general_new(
        "micro_recall",
        file_folder, threshold_list, lr_list, batchsize_list, epoch_list)
    print("micro_recall_averages: ", main_role_accuracy_averages)
    print("max_micro_recall_keys: ", max_main_role_accuracy_keys)
    print("max_micro_recall: ", max_main_role_accuracy)

    heat_map_lr_batchsize(main_role_accuracy_averages, threshold_list, lr_list, batchsize_list, epoch_list,"micro-recall", dir_name)



    dict = get_most_frequent_roles_new(file_folder, threshold_list, lr_list, batchsize_list, epoch_list)
    heat_map_max_role_frequency_lr_batchsize_new(dict, threshold_list, lr_list, batchsize_list,epochs, "most frequent classes", dir_name)'''



    
