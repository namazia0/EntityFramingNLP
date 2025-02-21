import pandas as pd
from dotenv import load_dotenv
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string

load_dotenv(dotenv_path="config/config.env")

def load_data(file_path):
    """
    Combine the annotations and the articles into one dataframe.
    """
    # file_path = f"./dataset/{mode}_4_december/{language}/subtask-1-annotations.txt"
    # article_path = f"./dataset/{mode}_4_december/{language}/raw-documents/"
    article_path = file_path.replace("subtask-1-annotations.txt", "raw-documents/")

    #df = pd.read_csv(file_path, delimiter="\t")

    df = pd.DataFrame(columns = ["article_id", "entity_mention", "start_offset", "end_offset", "main_role", "fine-grained_roles_1",
                  "fine-grained_roles_2"])

    print(df.head())
    print(len(df))

    with open(file_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 7:
                parts.append("")
            print("parts: ",parts)
            df.loc[line_num] = parts[:7]

    print("df: ",df)

    file_path_articles = article_path
    all_articles = df.iloc[:, 0]
    new_coloumb = []

    for article in all_articles:
        content = ""
        with open(file_path_articles + str(article), "r", encoding="utf-8") as file:
            content = file.read()
        new_coloumb.append(content)
    df["article_content"] = new_coloumb

    df.to_csv("combined_training_data.csv", index=False)

    return df

def load_data_2(file_path):
    """
    Combine the annotations and the articles into one dataframe.
    """
    # file_path = f"./dataset/{mode}_4_december/{language}/subtask-1-annotations.txt"
    # article_path = f"./dataset/{mode}_4_december/{language}/raw-documents/"
    article_path = file_path.replace("subtask-1-annotations.txt", "raw-documents/")

    df = pd.read_csv(file_path, delimiter="\t", header=None, names=range(8))

    df = df.rename(columns=lambda x: None)

    couloumb=[]
    for index, row in df.iterrows():
        temp = []
        for item in df.iloc[index, 5:]:
            temp.append(item)
        print("temp: ", temp)
        couloumb.append(temp)

    df.insert(5, "fine_grained_roles", couloumb)
    df = df.iloc[:, 0:6]

    print("df: ",df)

    df.columns = ["article_id", "entity_mention", "start_offset", "end_offset", "main_role", "fine_grained_roles"]

    file_path_articles = article_path
    all_articles = df.iloc[:, 0]
    new_coloumb = []

    for article in all_articles:
        content = ""
        with open(file_path_articles + str(article), "r", encoding="utf-8") as file:
            content = file.read()
        new_coloumb.append(content)
    #df["article_content"] = new_coloumb
    df['article_content'] = new_coloumb

    df.to_csv("combined_training_data_test.csv", index=False)

    print(df.head())

    return df

def sub_role_to_main_role(sub_role:str):
    """
    Map the sub-role to the main role
    """
    match sub_role:
        #Protagonist
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

        #Antagonist
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

        #Innocent
        case "Forgotten":
            return "Innocent"
        case "Exploited":
            return "Innocent"
        case "Victim":
            return "Innocent"
        case "Scapegoat":
            return "Innocent"

    return "NaN"


class Dataset():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):  
        # List all txt files in the directory
        txt_files = [f for f in os.listdir(self.dataset_path)]
        # Read the content of each file and store it in a list
        data = []
        for file_name in txt_files:
            with open(os.path.join(self.dataset_path, file_name), 'r') as file:
                text = file.read()
                data.append({'file_name': file_name, 'content': text})
        self.df = pd.DataFrame(data, columns=['file_name', 'content'])
        return self.df

    def calculate_sentence_length(self, text):
        sentences = sent_tokenize(text) 
        avg_sentence_length = sum(len(sentence.split(" ")) for sentence in sentences) / len(sentences)
        return avg_sentence_length

    def calculate_word_count(self, text):
        word_count = len(text.split(" ")) 
        return word_count

    def calculate_vocabulary_size(self, text):
        """
        Remove punctuation and convert to lower case.
        """
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        word_count = text.split(" ")
        vocabulary_size = set(word_count)
        return len(vocabulary_size)
    
    def remove_new_line(self, content):
        content = content.replace("\n", " ")
        return content.replace("\n\n", " ")
    
    def remove_s_in_content(self, content):
        content = content.split(" ")
        temp = []
        for word in content:
            if "s" == word:
                word = word.replace("s", "")
            else:
                temp.append(word)
        return temp
    
    def preprocess_data(self):
        """
        Apply all the function to each document in the dataset.
        """
        self.df["avg_sentence_length"] = self.df["content"].apply(self.calculate_sentence_length)
        self.df['word_count'] = self.df['content'].apply(self.calculate_word_count)
        self.df['vocabulary_size'] = self.df['content'].apply(self.calculate_vocabulary_size)
        self.df['content'] = self.df['content'].apply(self.remove_new_line)
        self.df['content'] = self.df['content'].apply(self.remove_s_in_content)
        return self.df
    
    def get_average_length_all_docs(self):
        return self.df['avg_sentence_length'].mean()
    
    def get_average_length_word_count_all_docs(self):
        return self.df['word_count'].mean()
    
    def get_average_length_vocabulary_size_all_docs(self):
        return self.df['vocabulary_size'].mean()
    
    def get_top_ten_documents(self):
        self.df['first_sentence'] = self.df['content'].apply(lambda x: sent_tokenize(x)[0])
        self.df = self.df.sort_values(by='word_count', ascending=False).head(10).reset_index(drop=True)
        return self.df
    
    def get_df(self):
        return self.df

if __name__ == '__main__':
    df = load_data_2("C:\\Users\hamml\PycharmProjects\entity-framing-nlp\dataset\\train_4_december\EN\subtask-1-annotations.txt")

    for item in df["fine_grained_roles"]:
        print("item: ",item)