from .base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from copy import copy
import numpy as np
from transformers import Trainer, TrainingArguments
# from datasets import load_dataset

from scripts.dataset import sub_role_to_main_role

from dotenv import load_dotenv

load_dotenv(dotenv_path="config/config.env")

# Input:
# News article text.
# Named entities (NEs) and their positions.

# Output:
# For each NE, assign one or more roles from the predefined taxonomy (and a confidence score).



class RoBERTa(nn.Module):
    def __init__(self, hyperparamters ,num_labels=22):
        super(RoBERTa, self).__init__()
        # was roberta-base
        self.model = RobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
        #self.sigmoid = nn.Sigmoid()  # Sigmoid for classification

        self.label_encoder = LabelEncoder()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.tokenized_dataset = None
        self.train_test_split = None

        #self.learning_rate = float(os.getenv("LR"))
        #self.batch_size = int(os.getenv("BATCH_SIZE"))
        #self.epochs = int(os.getenv("EPOCHS"))
        #self.weight_decay = float(os.getenv("WEIGHT_DECAY"))


        self.learning_rate = float(hyperparamters[0])
        self.batch_size = int(hyperparamters[1])
        self.epochs = int(hyperparamters[2])
        self.weight_decay = float(hyperparamters[3])
        self.iteration = int(hyperparamters[4])

    def forward(self, input_ids, attention_mask, labels=None):
        # Modellvorhersagen
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Sigmoid anwenden
        #probs = self.sigmoid(logits)

        if labels is not None:
            #print("labels not None")
            # Binary Cross-Entropy
            bce_loss = nn.BCEWithLogitsLoss()
            loss = bce_loss(logits, labels.float())
            positive_weight = 10.0  # weight is ten bc there can only be 2 / 22 classes true --> otherweise bias
            negative_weight = 1.0

            positive_loss = loss * labels * positive_weight
            negative_loss = loss * (1 - labels) * negative_weight


            combined_loss = positive_loss.mean() + negative_loss.mean()
            return combined_loss, logits

        return logits

    # add a second sub-role if it is not smaller than the relative threshold (>= highest_predicted_role_softmax - threhold) and if the sub-role is in the same main-role as the highest one
    # experiment with hyperparameter from init and environment variables via grid-search
    def predict(self, trainer, model,tokenizer , dev, threshold_list, result_save_name="predictions_with_labels.csv"):
        """
        Predict the roles and the sub-roles for the named entities in the dev file.

        Parameters:
        - dev_file: The news article dev file
        - named_entities: A list of named entities and their positions (span)

        Returns:
        - A list of predicted roles and sub-roles for each named entity
        """

        #model = RobertaForSequenceClassification.from_pretrained("./" + model_save_name)
        #tokenizer = RobertaTokenizer.from_pretrained("./" + model_save_name)

        if dev == 1:

            predictions = trainer.predict(self.tokenized_dataset["eval"])
        else:

            # predict test_old-data
            predictions = trainer.predict(self.tokenized_dataset["test"])

        logits = predictions.predictions

        true_classes = predictions.label_ids

        all_classes = self.label_encoder.classes_

        # for getting all ground_truth sub_roles in its binary vectorrepresentation e.g. [0, 1, 0, ..., 0, 1]
        def find_all_indices(lst, value):
            return [index for index, elem in enumerate(lst) if elem == value]

        results_df = pd.DataFrame()
        print("results_df: ", results_df)


        for i, sub_role in enumerate(all_classes):
            results_df[sub_role] = np.array(logits)[:, i]

        tc_1 = []
        tc_2 = []

        for true_class in true_classes:
            indices = find_all_indices(true_class, 1.)
            if len(indices) == 1:
                tc_1.append(all_classes[indices[0]])
                tc_2.append("NaN")
            else:
                tc_1.append(all_classes[indices[0]])
                tc_2.append(all_classes[indices[1]])

        results_df["fine_grained_roles_1_ground_truth"] = tc_1
        results_df["fine_grained_roles_2_ground_truth"] = tc_2


        def sub_role_to_main_role(sub_role: str):
            match sub_role:
                # Protagonist
                case "Guardian":
                    return "Protagonist"
                case "Martyr":
                    return "Protagonist"
                case "Peacemaker":
                    return "Protagonist"
                case "Rebel":
                    return "Protagonist"
                case "Underdog":
                    return "Protagonist"
                case "Virtuous":
                    return "Protagonist"

                # Antagonist
                case "Instigator":
                    return "Antagonist"
                case "Conspirator":
                    return "Antagonist"
                case "Tyrant":
                    return "Antagonist"
                case "Foreign Adversary":
                    return "Antagonist"
                case "Traitor":
                    return "Antagonist"
                case "Spy":
                    return "Antagonist"
                case "Saboteur":
                    return "Antagonist"
                case "Corrupt":
                    return "Antagonist"
                case "Incompetent":
                    return "Antagonist"
                case "Terrorist":
                    return "Antagonist"
                case "Deceiver":
                    return "Antagonist"
                case "Bigot":
                    return "Antagonist"

                # Innocent
                case "Forgotten":
                    return "Innocent"
                case "Exploited":
                    return "Innocent"
                case "Victim":
                    return "Innocent"
                case "Scapegoat":
                    return "Innocent"

            return "NaN"

        # get highest likely sub_class from predictions
        fine_grained_roles_1_predictions = []
        for index, line in results_df.iterrows():
            # 0:22 are the numeric sigmoid-predctions of all subclasses
            class_prediction = all_classes[np.argmax(line[0:22])]
            fine_grained_roles_1_predictions.append(class_prediction)

        results_df["fine_grained_roles_1_predictions"] = fine_grained_roles_1_predictions

        # ground-truth main-role
        results_df["main_role"] = self.train_test_split["test"]["main_role"]
        roles = []

        # predicted main-role
        for name in results_df["fine_grained_roles_1_predictions"]:
            roles.append(sub_role_to_main_role(name))
        results_df["main_role_prediction"] = roles

        #dict for evaluation_class_format

        def convert_df_to_right_format(test_df):
            #["article_id", "entity_mention", "start_offset", "end_offset", "main_role", "fine_grained_roles"]
            test_df = pd.DataFrame(test_df)
            formated_df = pd.DataFrame()
            formated_df["article_id"] = test_df["article_id"]
            formated_df["entity_mention"] = test_df["entity_mention"]
            formated_df["start_offset"] = test_df["start_offset"]
            formated_df["end_offset"] = test_df["end_offset"]
            formated_df["main_role"] = test_df["main_role"]
            formated_df["fine_grained_roles_1"] = test_df["fine-grained_roles_1"]
            formated_df["fine_grained_roles_2"] = test_df["fine-grained_roles_2"]

            for index, line in test_df.iterrows():
                if test_df["fine-grained_roles_2"][index] == None:
                    formated_df["fine_grained_roles_2"][index] = "NaN"

            formated_df.to_csv(result_save_name + "/" + "gold_labels_lr_" + str(self.learning_rate) + "_epochs_" + str(self.epochs) + "_batchsize_" + str(self.batch_size) + "_iteration_"+ str(self.iteration) +".tsv", sep='\t', index=False,
                              encoding='utf-8')

            return formated_df

        formated_df = convert_df_to_right_format(self.tokenized_dataset["test"])

        for threshold in threshold_list:

            second_highest_values = []
            second_highest_classes = []

            for index, line in results_df.iterrows():

                output = []
                for i in range(0, 22):
                    output.append(line[i])

                sorted_indices = np.argsort(output)[:: -1]
                sorted_outputs = np.array(output)[sorted_indices]

                for j in range(1, len(sorted_outputs)):
                    second_highest_value = sorted_outputs[j]
                    second_highest_class = sorted_indices[j]

                    if j == 1:
                        second_highest_values.append(second_highest_value)
                        second_highest_classes.append(all_classes[second_highest_class])
                    else:
                        second_highest_values[-1] = second_highest_value
                        second_highest_classes[-1] = second_highest_class

                    if sorted_outputs[0] - second_highest_value > threshold:
                        second_highest_classes[-1] = "NaN"
                        break

                    elif sub_role_to_main_role(second_highest_classes[-1]) != sub_role_to_main_role(
                            results_df["fine_grained_roles_1_predictions"][index]):
                        second_highest_classes[-1] = "NaN"


                    else:
                        break

            results_df["fine_grained_roles_2_predictions"] = second_highest_classes

            formated_df["fine_grained_roles_1"] = results_df["fine_grained_roles_1_predictions"]
            formated_df["fine_grained_roles_2"] = results_df["fine_grained_roles_2_predictions"]
            formated_df["main_role"] = results_df["main_role_prediction"]

            #save results and output of model
            formated_df.to_csv(result_save_name + "/" + result_save_name + "_threshold_" + str(threshold) + "_lr_" + str(self.learning_rate) + "_epochs_" + str(self.epochs) + "_batchsize_" + str(self.batch_size) + "_iteration_"+ str(self.iteration) + ".tsv", sep='\t', index=False, encoding='utf-8')
            results_df.to_csv(result_save_name + "/"+ "outputs_of_" + result_save_name + "_threshold_" + str(threshold) + "_lr_" + str(self.learning_rate) + "_epochs_" + str(self.epochs) + "_batchsize_" + str(self.batch_size) + "_iteration_"+ str(self.iteration) + ".tsv",
                               sep='\t', index=False, encoding='utf-8')

        return

def train(model, df,df_dev, model_save_name, result_save_name, thredhold_list,dynamic_seed):
    """
    Train the RoBERTa model on the training data.

    Parameters:
    - df: The training data as a pandas DataFrame

    Returns:
    - The trained RoBERTa model
    """


    # Define roles and sub-roles for random guessing
    # Add named_entities and roles und subroles lists as a parameter in the prediction?? Use the extract_named_entities function to get the named entities from the dev file
    # ROLES = ['Protagonist', 'Antagonist', 'Innocent']
    # PROTAGONISTS = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous']
    # ANTAGONISTS = ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot']
    # INNOCENTS = ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']

    df["labels"] = model.label_encoder.fit_transform(df["fine-grained_roles_1"])

    def convert_labels_to_multilabel_format(fine_grained_role_1, fine_grained_role_2,num_labels=22):

        # Erstelle einen Tensor mit der Größe (1, num_labels), initialisiert mit Nullen
        multilabel_tensor = torch.zeros(num_labels)

        # Setze die entsprechenden Positionen auf 1
        if fine_grained_role_1 in model.label_encoder.classes_:
            multilabel_tensor[list(model.label_encoder.classes_).index(fine_grained_role_1)] = 1

        if fine_grained_role_2 in model.label_encoder.classes_:
            multilabel_tensor[list(model.label_encoder.classes_).index(fine_grained_role_2)] = 1

        return multilabel_tensor



    # encode labels
    # old: df["labels"] = self.label_encoder.fit_transform(df["fine-grained_roles_1"])
    # new:
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda x: {
            'labels': convert_labels_to_multilabel_format(x['fine-grained_roles_1'], x['fine-grained_roles_2'])},
        batched=False
    )

    '''dataset_dev = Dataset.from_pandas(df_dev)
    dataset_dev = dataset.map(
        lambda x: {
            'labels': convert_labels_to_multilabel_format(x['fine-grained_roles_1'], x['fine-grained_roles_2'])},
        batched=False
    )'''

    dataset = dataset.shuffle(seed=dynamic_seed)
    model.train_test_split = dataset.train_test_split(test_size=0.2)
    #print("model.train_test_split: ", model.train_test_split["train"]["article_id"])

    def preprocess_data(examples):

        # combine article_content with entity_mention
        combined_texts = []
        for article, entity, start, end in zip(examples["article_content"], examples["entity_mention"], examples["start_offset"], examples["end_offset"]):
            combined_text = article + " [ENTITY] " + entity + " [START_OFFSET] " + str(start) + " [END_OFFSET] " + str(end)
            combined_texts.append(combined_text)

        # Tokenize combined text
        inputs = model.tokenizer(combined_texts, padding="max_length", truncation=True)

        # adding labels
        inputs["labels"] = examples["labels"]  # Labels hinzufügen

        return inputs


    model.tokenized_dataset = model.train_test_split.map(preprocess_data, batched=True)
    #model.tokenized_dataset["train"] = model.dataset.map(preprocess_data, batched=True)
    #model.tokenized_dataset["eval"] = model.dataset_dev.map(preprocess_data, batched=True)
    #print(model.tokenized_dataset)
    #print("model.train_test_split 2: ", model.train_test_split["train"]["article_id"])
    print("model.tokenized_dataset[train]: ", model.tokenized_dataset["train"])
    # num_labels = len(df["labels"].unique())
    # num_labels = 23
    # print("number of classes: ", num_labels) # number of classes

    # model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    # preparing training
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=model.learning_rate,
        per_device_train_batch_size=model.batch_size,
        num_train_epochs=model.epochs,
        weight_decay=model.batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=model.tokenized_dataset["train"],
        eval_dataset=model.tokenized_dataset["test"],
    )


    # start training
    trainer.train()

    # save model
    #model.model.save_pretrained("./" + model_save_name)
    #model.tokenizer.save_pretrained("./" + model_save_name)

    #loaded_model = RobertaForSequenceClassification.from_pretrained("./" + model_save_name)
    #tokenizer = RobertaTokenizer.from_pretrained("./" + model_save_name)


    #model.predict(trainer, model_save_name,0, thredhold_list, result_save_name=result_save_name)
    model.predict(trainer,model.model, model.tokenizer, 0, thredhold_list, result_save_name=result_save_name)


    return
