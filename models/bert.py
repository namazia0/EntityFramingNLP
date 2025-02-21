from .base_model import BaseModel
from scripts import loss  
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification 
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from copy import copy
import numpy as np
from transformers import Trainer, TrainingArguments
from scripts.dataset import sub_role_to_main_role  
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/config.env")

# This class defines a BERT-based model for named entity role classification.
# Input:
#     - News article text.
#     - Named entities (NEs) and their positions.
# Output:
#     - For each NE, assign one or more roles from a predefined taxonomy (with a confidence score).

class BERT(nn.Module):
    def __init__(self, hyperparameters, num_labels=22):
        super(BERT, self).__init__()
        # Load the pre-trained BERT model for sequence classification.
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

        # Initialize a label encoder for encoding/decoding role labels.
        self.label_encoder = LabelEncoder()
        # Load the BERT tokenizer.
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Initialize variables to store the tokenized dataset and train/test split.
        self.tokenized_dataset = None
        self.train_test_split = None

        # Hyperparameters, using the provided list
        self.learning_rate = float(hyperparameters[0])
        self.batch_size = int(hyperparameters[1])
        self.epochs = int(hyperparameters[2])
        self.weight_decay = float(hyperparameters[3])
        self.iteration = int(hyperparameters[4])

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            labels: Optional labels for training.

        Returns:
            If labels are provided, returns the loss and logits. Otherwise, returns only the logits.
        """
        # Get model predictions.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is not None:
            # Calculate Binary Cross-Entropy loss with weighted classes for handling class imbalance.
            bce_loss = nn.BCEWithLogitsLoss()
            loss = bce_loss(logits, labels.float())
            # Assign higher weight to positive examples (minority class).
            positive_weight = 10.0
            negative_weight = 1.0

            positive_loss = loss * labels * positive_weight
            negative_loss = loss * (1 - labels) * negative_weight

            combined_loss = positive_loss.mean() + negative_loss.mean()
            return combined_loss, logits

        return logits

    def predict(self, trainer, model, tokenizer, dev_file, threshold_list, result_save_name="predictions_with_labels.csv"):
        """
        Predicts the roles and sub-roles for named entities in a given dev file.

        Args:
            trainer: The Hugging Face Trainer object.
            model: The trained BERT model.
            tokenizer: The BERT tokenizer.
            dev_file: Path to the development data file (not directly used here, could be used for loading data).
            threshold_list: List of thresholds to consider for assigning a second sub-role.
            result_save_name: Base name for saving prediction results.
        """

        # Make predictions on the test set.
        predictions = trainer.predict(self.tokenized_dataset["test"])
        logits = predictions.predictions
        true_classes = predictions.label_ids
        all_classes = self.label_encoder.classes_

        def find_all_indices(lst, value):
            """Helper function to find all indices of a value in a list."""
            return [index for index, elem in enumerate(lst) if elem == value]

        results_df = pd.DataFrame()

        # Store the prediction scores for each sub-role in the DataFrame.
        for i, sub_role in enumerate(all_classes):
            results_df[sub_role] = np.array(logits)[:, i]

        # Extract the ground truth sub-roles from the true_classes.
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

        # Predict the primary fine-grained role based on the highest prediction score.
        fine_grained_roles_1_predictions = []
        for index, line in results_df.iterrows():
            class_prediction = all_classes[np.argmax(line[0:22])]
            fine_grained_roles_1_predictions.append(class_prediction)

        results_df["fine_grained_roles_1_predictions"] = fine_grained_roles_1_predictions

        # Add the ground truth main role.
        results_df["main_role"] = self.train_test_split["test"]["main_role"]

        # Predict the main role based on the predicted primary fine-grained role.
        roles = []
        for name in results_df["fine_grained_roles_1_predictions"]:
            roles.append(sub_role_to_main_role(name))
        results_df["main_role_prediction"] = roles

        def convert_df_to_right_format(test_df):
            """
            Converts the test data into a format suitable for evaluation and saves gold labels.
            """
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
                if test_df["fine-grained_roles_2"][index] is None:
                    formated_df["fine_grained_roles_2"][index] = "NaN"

            formated_df.to_csv("predictions_results/" + "gold_labels_lr_" + str(self.learning_rate) + "_batchsize_" + str(self.batch_size) + "_iteration_"+ str(self.iteration) +".tsv", sep='\t', index=False, encoding='utf-8')
            return formated_df

        formated_df = convert_df_to_right_format(self.tokenized_dataset["test"])

        # Iterate through different thresholds to determine if a second sub-role should be assigned.
        for threshold in threshold_list:
            second_highest_values = []
            second_highest_classes = []

            for index, line in results_df.iterrows():
                output = [line[i] for i in range(0, 22)]
                sorted_indices = np.argsort(output)[::-1]
                sorted_outputs = np.array(output)[sorted_indices]

                for j in range(1, len(sorted_outputs)):
                    second_highest_value = sorted_outputs[j]
                    second_highest_class = sorted_indices[j]

                    if j == 1:
                        second_highest_values.append(second_highest_value)
                        second_highest_classes.append(all_classes[second_highest_class])
                    else:
                        second_highest_values[-1] = second_highest_value
                        second_highest_classes[-1] = all_classes[second_highest_class]
                    
                    # Assign NaN if the difference between the top two predictions is greater than the threshold.
                    # Also assign NaN if the second highest prediction belongs to a different main role.
                    if sorted_outputs[0] - second_highest_value > threshold:
                        second_highest_classes[-1] = "NaN"
                        break
                    elif sub_role_to_main_role(second_highest_classes[-1]) != sub_role_to_main_role(results_df["fine_grained_roles_1_predictions"][index]):
                        second_highest_classes[-1] = "NaN"
                    else:
                        break

            results_df["fine_grained_roles_2_predictions"] = second_highest_classes

            formated_df["fine_grained_roles_1"] = results_df["fine_grained_roles_1_predictions"]
            formated_df["fine_grained_roles_2"] = results_df["fine_grained_roles_2_predictions"]
            formated_df["main_role"] = results_df["main_role_prediction"]

            # Save the prediction results and model outputs.
            formated_df.to_csv("predictions_results/" + result_save_name + "_threshold_" + str(threshold) + "_lr_" + str(self.learning_rate) + "_batchsize_" + str(self.batch_size) + "_iteration_"+ str(self.iteration) + ".tsv", sep='\t', index=False, encoding='utf-8')
            results_df.to_csv("predictions_results/" + "outputs_of_" + result_save_name + "_threshold_" + str(threshold) + "_lr_" + str(self.learning_rate) + "_batchsize_" + str(self.batch_size) + "_iteration_"+ str(self.iteration) + ".tsv", sep='\t', index=False, encoding='utf-8')

        return

def train(model, df, model_save_name, result_save_name, threshold_list):
    """
    Trains the BERT model on the provided data.

    Args:
        model: The BERT model instance.
        df: The training data as a pandas DataFrame.
        model_save_name: Name for saving the trained model (not used in the current implementation).
        result_save_name: Base name for saving prediction results.
        threshold_list: List of thresholds to use during prediction.
    """

    # Encode the fine-grained roles using the label encoder.
    df["labels"] = model.label_encoder.fit_transform(df["fine_grained_roles_1"])

    def convert_labels_to_multilabel_format(fine_grained_role_1, fine_grained_role_2, num_labels=22):
        """Converts the fine-grained roles into a multi-label format."""
        multilabel_tensor = torch.zeros(num_labels)
        if fine_grained_role_1 in model.label_encoder.classes_:
            multilabel_tensor[list(model.label_encoder.classes_).index(fine_grained_role_1)] = 1
        if fine_grained_role_2 in model.label_encoder.classes_:
            multilabel_tensor[list(model.label_encoder.classes_).index(fine_grained_role_2)] = 1
        return multilabel_tensor

    # Convert the DataFrame to a Hugging Face Dataset.
    dataset = Dataset.from_pandas(df)
    # Convert the fine-grained roles to multi-label format.
    dataset = dataset.map(
        lambda x: {
            'labels': convert_labels_to_multilabel_format(x['fine-grained_roles_1'], x['fine-grained_roles_2'])},
        batched=False
    )

    # Shuffle and split the dataset into training and testing sets.
    dataset = dataset.shuffle(seed=42)
    model.train_test_split = dataset.train_test_split(test_size=0.2)

    def preprocess_data(examples):
        """Preprocesses the data by combining article content with entity mentions and tokenizing."""
        combined_texts = []
        for article, entity, start, end in zip(examples["article_content"], examples["entity_mention"], examples["start_offset"], examples["end_offset"]):
            combined_text = article + " [ENTITY] " + entity + " [START_OFFSET] " + str(start) + " [END_OFFSET] " + str(end)
            combined_texts.append(combined_text)

        # Tokenize the combined text.
        inputs = model.tokenizer(combined_texts, padding="max_length", truncation=True)
        inputs["labels"] = examples["labels"]
        return inputs

    # Tokenize the dataset.
    model.tokenized_dataset = model.train_test_split.map(preprocess_data, batched=True)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=model.learning_rate,
        per_device_train_batch_size=model.batch_size,
        num_train_epochs=model.epochs,
        weight_decay=model.weight_decay, # Corrected this line to use the model's weight decay
    )

    # Create a Trainer instance.
    trainer = Trainer(
        model=model.model,  # Pass the model attribute of the BERT class
        args=training_args,
        train_dataset=model.tokenized_dataset["train"],
        eval_dataset=model.tokenized_dataset["test"],
    )

    # Start training.
    trainer.train()

    # Make predictions using the trained model.
    model.predict(trainer, model.model, model.tokenizer, 0, threshold_list, result_save_name=result_save_name)

    return
