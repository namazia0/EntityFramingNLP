import csv, random, os
from collections import Counter
import argparse
import datetime
import scorer
import pandas as pd

from models.train import train
from models.test import test

from scripts.dataset import load_data, load_data_2
from models import get_model, RoBERTa

# Set a fixed seed for reproducibility of random operations

from models.roberta_new import RoBERTa
from models.roberta_new import train

# Define roles and sub-roles for random guessing
ROLES = ['Protagonist', 'Antagonist', 'Innocent']
PROTAGONISTS = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous']
ANTAGONISTS = ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot']
INNOCENTS = ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']

def process_tsv_file(file_path):
    """
    Reads a TSV file as a single string and manually processes it based on tabs and newlines.

    Parameters:
    - file_path: Path to the TSV file.

    Returns:
    - A list of rows, where each row is a list of columns.
    """
    try:
        # Read the file as a single string
        with open(file_path, mode='r', encoding='utf-8') as file:
            file_content = file.read()

        # Split content into lines
        lines = file_content.split('\n')

        # Process each line into columns based on tabs
        rows = []
        for line in lines:
            if line.strip():  # Skip empty lines
                columns = line.split('\t')
                rows.append(columns)

        return rows
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return None
    
def random_guess():
    """
    Assigns a random role and sub-role to an entity.
    
    Returns:
    - A tuple containing the randomly chosen role and sub-role
    """    
    role = random.choice(ROLES)
    if role == 'Protagonist':
        sub_role = random.choice(PROTAGONISTS)
    elif role == 'Antagonist':
        sub_role = random.choice(ANTAGONISTS)
    else:
        sub_role = random.choice(INNOCENTS)
    return role, sub_role

def majority_votes(train_file):
    """
    Assigns the most common role and sub-role based on training data.
    
    Parameters:
    - train_file: Path to the training data file
    
    Returns:
    - A tuple containing the most common role and sub-role from the training file
    """
    # Load roles and sub-roles from the training file to find the most common values    
    rows = process_tsv_file(train_file)

    roles = []
    sub_roles = []
    for row in rows:
        roles.append(row[4])
        sub_roles.append(row[5])
    # Return the most common role and sub-role
    return Counter(roles).most_common(1)[0][0], Counter(sub_roles).most_common(1)[0][0]

def main(dev_file, output_dir, model_name, train_file=None):
    """
    Processes the input file to generate baseline predictions for entity roles.
    
    Parameters:
    - dev_file / test_old file: Path to the dev / test_old file containing entities and the offsets
    - output_dir: Directory to save the output file
    - model_name: Type of baseline to use, default roberta
    - train_file: Optional path to the training file, required if using an architecture for training
    """

    mode = os.getenv("MODE")                # TRAIN, DEV, TEST
    language = os.getenv("LANGUAGE")        # EN or PT

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_train = load_data_2(train_file)
    print("df_train: ",df_train)
    df_dev = 0
    
    # Set output file name based on baseline type
    time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    output_file = os.path.join(output_dir, f"baseline_{model_name}_{time}.txt")

    # Precompute role and sub-role
    if train_file is None:
        print("ERROR: train_file is required for the architectures.")
        return
    if model_name == "majority":
        role, sub_role = majority_votes(train_file)
        print(f"Majority baseline: Role={role}, Sub-role={sub_role}")
    if model_name == "random":
        role, sub_role = random_guess()
        print(f"Majority baseline: Role={role}, Sub-role={sub_role}")

    try:
        with open(dev_file, mode='r', encoding='utf-8') as file:
            with open(output_file, mode='w', encoding='utf-8') as fout:
                tsv_reader = csv.reader(file, delimiter='\t')
                writer = csv.writer(fout, delimiter='\t')
                for i, row in enumerate(tsv_reader):
                    # Extract necessary fields from each row
                    file_id = row[0]
                    obj = row[1]
                    span_start = row[2]
                    span_end = row[3]

                    # Generate role and sub-role for random baseline
                    if model_name == "random":
                        role, sub_role = random_guess()
                        print(f"Random baseline: Role={role}, Sub-role={sub_role}")

                    # Write the predicted role and sub-role to the output file
                    writer.writerow([file_id, obj, span_start, span_end, role[i], sub_role[i]])
        return
    except FileNotFoundError:
        print(f"ERROR: File not found '{dev_file}'")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")

    print("done!")

    # Train the model with training data because we already have the labels there and then apply the model on dev test_old.
    threshold_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.5, 0.7]#[0.05, 0.075, 0.1]#[0.0005, 0.001, 0.005, 0.01, 0.05]
    lr_list = [4e-5]#[3e-5, 4e-5, 5e-5, 6e-5]#[8e-5, 1e-4, 1e-3, 1e-2]#[4e-5, 5e-5, 6e-5]#[1e-5, 2e-5, 3e-5]
    batchsize_list = [8]
    epochs_list = [20]#[6, 10, 15, 20]#[3]#[6, 7, 8]
    weight_decay = 0.01
    model_save_name = "my_xml_roberta_model"
    result_save_name = "roberta_PT_EN"

    print("df_train: ", df_train)

    if not os.path.exists(result_save_name):
        os.makedirs(result_save_name)
    dynamic_seed = 0
    for lr in lr_list:
        for epochs in epochs_list:
            for batchsize in batchsize_list:
                for iteration in range(0,10):
                    dynamic_seed += 1
                    print("dynamic_seed: ", dynamic_seed)
                    hyperparamters = [lr, batchsize, epochs, weight_decay, iteration]
                    model = RoBERTa(hyperparamters)
                    train(model, df_train,df_dev, model_save_name, result_save_name, threshold_list, dynamic_seed)
                    #model, df,df_dev, model_save_name, result_save_name, thredhold_list,dynamic_seed
                    metric_score_df = pd.DataFrame()
                    metric_score_df["threshold"] = threshold_list
                    for threshold in threshold_list:
                        emr, micro_precision, micro_recall, micro_f1, main_role_accuracy = scorer.main(result_save_name + "/" + "gold_labels_lr_" + str(lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_"+ str(iteration) +".tsv", result_save_name + "/" + result_save_name + "_threshold_" + str(threshold) +"_lr_" + str(lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_"+ str(iteration) + ".tsv")
                        metric_score_df["emr"] = emr
                        metric_score_df["micro_precision"] = micro_precision
                        metric_score_df["micro_recall"] = micro_recall
                        metric_score_df["micro_f1"] = micro_f1
                        metric_score_df["main_role_accuracy"] = main_role_accuracy

                    metric_score_df.to_csv(result_save_name + "/metric_scores_lr_" + str(lr) + "_epochs_" + str(epochs) + "_batchsize_" + str(batchsize) + "_iteration_" + str(iteration) +".csv", index=False)

if __name__ == "__main__":
    # Argument parsing for command-line inputs
    parser = argparse.ArgumentParser(description="Run baselines for role annotation")
    parser.add_argument('--dev_file', type=str, required=True, help="Path to the input test_old file to predict the roles and sub-roles for the given named entites.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output file in the corresponding model and language.")
    parser.add_argument('--model_name', type=str, choices=['random', 'majority', 'roberta', 'bert', 'gpt', 'xlnet', 'roberta_NN'], required=True, help="Architecture: default roberta_NN")
    parser.add_argument('--train_file', type=str, help="Path to the training file (required for the four architectures during training)")

    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args.dev_file, args.output_dir, args.model_name, args.train_file)