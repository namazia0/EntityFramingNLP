from scripts.visualizer import boxplot
from llm_scorer import main as scorer
from dotenv import load_dotenv
from prompts import Prompts
import numpy as np
import csv
import os
import re

# Load environment variables from the specified .env file
load_dotenv(dotenv_path="config/config.env")

"""
File for predicting the roles using the extracted sentence and the additional created context.
"""

class ProcessArticle():
    def extract_sentence(self, file_path, entity, start_offset, end_offset):
        """
        Extract Sentence where the entity appears in the article text regarding its offset
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        text = ''.join(lines)

        # Find sentences using a basic sentence splitter (e.g., punctuation like ".", "!", "?")
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        start = start_offset
        end = end_offset 
        entity_range = range(start, end)

        current_pos = 0
        for sentence in sentences:
            sentence_start = current_pos
            sentence_end = current_pos + len(sentence)

            # Check if the entity offset lies within this sentence's range
            if sentence_start <= start < sentence_end and sentence_start <= end <= sentence_end:
                # Double-check the entity is present in this sentence
                if entity in sentence:
                    return sentence.strip()

            # Update current position for the next sentence
            current_pos = sentence_end + 1  # Account for splitting spaces or punctuation

        return None  # Return None if no matching sentence is found

    def get_document(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Skip the first line
        lines_without_first = lines[1:]
        
        # Join the remaining lines into a single string and remove all newline characters
        content = ''.join(lines_without_first).replace('\n', '')
        
        return content

    def get_title(self, file_path):
        """
        Get the title of the document (first line)
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            title = file.readline().strip()
        return title

    def extract_entites(self, txt_file):
        """
        Extract named entities from a given file
        """
        named_entities = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            for line in file:
                row = line.strip().split('\t')
                if len(row) >= 4:
                    file_name = row[0]
                    entity = row[1]
                    start_offset = int(row[2])
                    end_offset = int(row[3])
                    
                    named_entities.append({
                        "file_name": file_name,
                        "entity": entity,
                        "start_offset": start_offset,
                        "end_offset": end_offset
                    })
        return named_entities

def main():
    prompt = Prompts()
    process = ProcessArticle()

    language = os.getenv("LANGUAGE") # EN or PT (portuguese)     

    MAIN_ROLES = ['Antagonist', 'Protagonist', 'Innocent']
    FINE_GRAINED_ROLES = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous',
                        'Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot',
                        'Forgotten', 'Exploited', 'Victim', 'Scapegoat']

    output_file = "./prediction_llm_tests.txt"
    # val set
    dev_file = f"./dataset/dev_4_december/{language}/subtask-1-entity-mentions.txt"      
    # test set
    # test_file = "./EN_test/subtask-1-entity-mentions.txt"
    # entity_mentions = "./EN_test/subtask-1-entity-mentions.txt"

    accuracy_list = []
    micro_f1_list = []
    macro_f1_list = []
    exact_match_ratio_list = [] 

    iteration = 10

    # Extract entities from the development file
    entities = process.extract_entites(dev_file)

    for i in range(iteration):
        print(f"Iteration {i+1} running ...")
        role = []
        sub_role = []
        for entity in entities:
            # print(f"File: {entity['file_name']}, Entity: {entity['entity']}, Start: {entity['start_offset']}, End: {entity['end_offset']}")

            article_path = f"./dataset/dev_4_december/{language}/subtask-1-documents/{entity['file_name']}"     # validation set
            # article_path = f"./EN_test/subtask-1-documents/{entity['file_name']}"     # test set

            start_offset = entity['start_offset']
            end_offset = entity['end_offset']

            # Extract the sentence containing the entity
            sentence = process.extract_sentence(article_path, entity['entity'], start_offset, end_offset)
            # Get the title of the document
            title = process.get_title(article_path)
            # Get the content of the document excluding the title
            document = process.get_document(article_path)

            # Create context if specified in the environment variables
            if int(os.getenv("CONTEXT")) == 1:
                context = prompt.create_context(title, document, sentence, entity['entity'])
            else: 
                context = ""
            # Generate prediction based on the sentence and context
            prediction = prompt.generate_prediction(sentence, context, entity['entity'], language)
            print("predicion: ", prediction)
            if int(os.getenv("IN_CONTEXT_LEARNING")) == 3:
                print(prediction)
                print("Prediction: ", prediction.strip().split("\n")[-1])

            parts = prediction.split(": ")
            key = parts[0].strip()
            print("key: ", key)
            if key not in MAIN_ROLES:
                key = "Antagonist"
            role.append(key)
            
            values = [v.strip() for v in parts[1].split(",")]  # Extract sub-roles as a list
            sub_role.append(values)

        try:
            with open(dev_file, mode='r', encoding='utf-8') as file:
                with open(output_file, mode='w', encoding='utf-8') as fout:
                    tsv_reader = csv.reader(file, delimiter='\t')
                    writer = csv.writer(fout, delimiter='\t')
                    for j, row in enumerate(tsv_reader):
                        file_id = row[0]
                        obj = row[1]
                        span_start = row[2]
                        span_end = row[3]
                        if len(sub_role[j]) > 1:
                            # Write the predicted role and sub-role to the output file
                            if sub_role[j][0] in FINE_GRAINED_ROLES and sub_role[j][1] in FINE_GRAINED_ROLES:
                                writer.writerow([file_id, obj, span_start, span_end, role[j], sub_role[j][0], sub_role[j][1]])
                            elif sub_role[j][0] in FINE_GRAINED_ROLES and sub_role[j][1] not in FINE_GRAINED_ROLES:
                                writer.writerow([file_id, obj, span_start, span_end, role[j], sub_role[j][0]])
                            elif sub_role[j][0] not in FINE_GRAINED_ROLES and sub_role[j][1] in FINE_GRAINED_ROLES:
                                writer.writerow([file_id, obj, span_start, span_end, role[j], sub_role[j][1]])
                            elif sub_role[j][0] not in FINE_GRAINED_ROLES and sub_role[j][1] not in FINE_GRAINED_ROLES:
                                writer.writerow([file_id, obj, span_start, span_end, role[j]])
                        else:
                            if sub_role[j][0] in FINE_GRAINED_ROLES:
                                writer.writerow([file_id, obj, span_start, span_end, role[j], sub_role[j][0]])
                            else:
                                writer.writerow([file_id, obj, span_start, span_end, role[j]])
        except FileNotFoundError:
            print(f"ERROR: File not found '{dev_file}'")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred: {e}")

        gold_file_path = f"./dataset/dev_4_december/{language}/subtask-1-annotations.txt"

        # Calculate evaluation metrics
        acc, micro_f1, macro_f1, emr = scorer(gold_file_path, output_file)
        print(f"Accuracy: {acc}, Micro-F1: {micro_f1}, Macro-F1: {macro_f1}, Exact Match Ratio: {emr}")
        micro_f1_list.append(round(micro_f1, 4))
        macro_f1_list.append(round(macro_f1, 4))
        accuracy_list.append(round(acc, 4))
        exact_match_ratio_list.append(round(emr, 4))

    # Calculate average metrics
    avg_micro_f1 = round(np.mean(micro_f1_list), 4)
    avg_macro_f1 = round(np.mean(macro_f1_list), 4)
    avg_acc = round(np.mean(accuracy_list), 4)
    avg_emr = round(np.mean(exact_match_ratio_list), 4)
    print("micro F1 list = ", micro_f1_list)
    print("macro F1 list = ", macro_f1_list)
    print("accuracy list = ", accuracy_list)
    print("exact match ratio list: ", exact_match_ratio_list)

    print(f"avg macro F1 = {avg_macro_f1}, avg micro F1 = {avg_micro_f1}, avg accuracy = {avg_acc}, avg emr = {avg_emr}")

    data = {
        "Accuracy": accuracy_list,
        "Micro F1": micro_f1_list,
        "Macro F1": macro_f1_list,
    }

    # Visualize the results
    boxplot(data)


if __name__ == "__main__":
    main()
