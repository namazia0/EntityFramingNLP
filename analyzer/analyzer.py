import random
import pandas as pd
from scripts.dataset import load_data
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    emr = []
    micro_precision = []
    avg_values_f1 = []
    micro_recall = []
    micro_f1 = []
    for thresholds in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.5, 0.7]:
        for name in ["emr", "micro_precision", "micro_recall", "micro_f1", "main_role_accuracy"]:
            avg_value = 0
            for iteration in range(10):
                df = pd.read_csv("roberta_PT/metric_scores_lr_4e-05_epochs_20_batchsize_8_iteration_" + str(iteration) + ".csv")

                for index,line in df.iterrows():
                    if df["threshold"][index] == thresholds:
                        avg_value += df[name][index]
            avg_value = avg_value / 10
            print(thresholds,", ",name,": ",avg_value)

    with open("dataset/combined/subtask-1-annotations-en.txt", "r",encoding="utf-8") as f1, open("dataset/combined/subtask-1-annotations-pt.txt", "r",encoding="utf-8") as f2:
        lines = f1.readlines() + f2.readlines()  # Combine the lines

    random.shuffle(lines)  # Shuffle the lines randomly

    # Write the shuffled lines to a new file
    with open("dataset/combined/subtask-1-annotations.txt", "w", encoding="utf-8") as out:
        out.writelines(lines)

    '''df_train = load_data("dataset/train_4_december/EN/subtask-1-annotations.txt")
    print(df_train.keys())
    protagonist = 0
    antagonist = 0
    innocent = 0
    for role in df_train["main_role"]:
        if role == "Protagonist":
            protagonist += 1
        elif role == "Antagonist":
            antagonist += 1
        else:
            innocent += 1

    # Data for the plot
    categories = ["Protagonist", "Antagonist", "Innocent"]
    values = [protagonist, antagonist, innocent]
    colors = ["#A3FFB3", "#FFB3B3", "#A3C8FF"]#["#1f77b4", "#ff7f0e", "#2ca02c"]  # Custom colors (blue, orange, green)

    # Set a modern style
    plt.style.use("fivethirtyeight")

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = plt.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)

    # Add gridlines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add title and labels
    plt.title("Distribution of Main Roles", fontsize=16, fontweight="bold")
    plt.ylabel("Number of Entities", fontsize=12)
    plt.xlabel("Roles", fontsize=12)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height}", ha="center", va="bottom", fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()
    '''
    ''' file_folder = "roberta_NN_final_results"
    #dir_name = "heatmaps_roberta_final_results"
    
    
    threshold_list = [0.05, 0.075, 0.1]  # [0.0005, 0.001, 0.005, 0.01, 0.05]
    lr_list = [1e-5, 2e-5, 3e-5]  # [8e-5, 1e-4, 1e-3, 1e-2]#[4e-5, 5e-5, 6e-5]#[1e-5, 2e-5, 3e-5]
    batchsize_list = [8, 16, 32]
    epochs_list = [3]  # [6, 7, 8]
    for lr in lr_list:
        for epochs in epochs_list:
            for batchsize in batchsize_list:
                for iteration in range(0,10):
                    data = pd.read_csv(file_folder + "/gold_labels_lr_" + str(lr) + "_epochs_" + str(
                        epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) + ".tsv", sep="\t")
                    for index, line in data.iterrows():
                        if data["fine_grained_roles_2"][index] == "NaN":
                            data["fine_grained_roles_2"][index] = ""
    
                    data.to_csv(file_folder + "/" + "gold_labels_lr_" + str(lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) + ".tsv", sep='\t', index=False,
                        encoding='utf-8')'''

"""    lr_list = [1e-5, 2e-5, 3e-5]  # [8e-5, 1e-4, 1e-3, 1e-2]#[4e-5, 5e-5, 6e-5]#[1e-5, 2e-5, 3e-5]
    batchsize_list = [8, 16, 32]
    epochs_list = [3]  # [6, 7, 8]
    threshold_list = [0.05, 0.075, 0.1]  # [0.0005, 0.001, 0.005, 0.01, 0.05]
    file_folder = "old_results/roberta_final_results"

    for lr in lr_list:
        for epochs in epochs_list:
            for batchsize in batchsize_list:
                for iteration in range(0,10):

                    metric_score_df = pd.DataFrame()
                    metric_score_df["threshold"] = threshold_list
                    for threshold in threshold_list:
                        emr, micro_precision, micro_recall, micro_f1, main_role_accuracy = scorer.main(file_folder + "/" + "gold_labels_lr_" + str(lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_"+ str(iteration) +".tsv", file_folder + "/" + file_folder + "_threshold_" + str(threshold) +"_lr_" + str(lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_"+ str(iteration) + ".tsv")
                        metric_score_df["emr"] = emr
                        metric_score_df["micro_precision"] = micro_precision
                        metric_score_df["micro_recall"] = micro_recall
                        metric_score_df["micro_f1"] = micro_f1
                        metric_score_df["main_role_accuracy"] = main_role_accuracy
                        print("micro_f1: ",micro_f1)

                    metric_score_df.to_csv(file_folder + "/metric_scores_lr_" + str(lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) +".csv", index=False)

                    print()"""