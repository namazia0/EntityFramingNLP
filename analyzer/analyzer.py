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