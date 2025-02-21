from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import os
import requests
import json
import yaml

load_dotenv(dotenv_path="config/config.env")

def tabby_api(prompt, max_tokens):
    tabby_api_url = "http://localhost:5001/v1"
    tabby_api_key = os.getenv("TABBY_API_KEY")
    # llm_model = "Llama-3.1-8B-Instruct-exl2"

    with open("../tabbyAPI/config.yml", "r") as file:
        config = yaml.safe_load(file)
    llm_model = config["model"]["model_name"]
    # print("llm model: ", llm_model)

    if not tabby_api_key:
        raise ValueError("API key is not set in the environment variable 'TABBY_API_KEY'.")

    url = f'{tabby_api_url}/chat/completions'
    headers = {
    'Content-Type': 'application/json',
    "Authorization": f'Bearer {tabby_api_key}'
    }

    payload = json.dumps({
    "model": llm_model,
    "max_tokens": max_tokens,
    "stream": "False",
    "min_p": 0.05,
    "messages": [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ]
            },
        ],
    "repetition_penalty": 1.05
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    response = json.loads(response.text)
    return response['choices'][0]['message']['content']


def translate(text):
    prompt = f"""Translate the text into French. Preserve all named entities exactly as they appear in the text.
    Do not add any additional information which do not appear in the text.
    Do not preamble.
    
    Text: {text} 
    Translation: 
    """

    max_tokens = len(text)

    translation = tabby_api(prompt, max_tokens)
    return translation

def backtranslate(text, target_language):
    prompt = f"""Translate the text into {target_language}. Preserve all named entities exactly as they appear in the text. 
    Do not add any additional information which do not appear in the text.
    Do not preamble.
    
    Text: {text} 
    Translation: 
    """

    max_tokens = len(text)

    translation = tabby_api(prompt, max_tokens)
    return translation

# Directory containing the files
directory = "./dataset/train_4_december/EN/raw-documents"
target_language = "English"
translated_path = "./translation/en/"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    name, ext = os.path.splitext(filename)

    # Check if it's a file and has a .txt extension
    if os.path.isfile(file_path) and filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            print(f"Content of {filename}:\n{content}\n\n")
            translated_text = translate(content)
            print(translated_text, "\n\n")
            final_translation = backtranslate(translated_text, target_language)
            print(final_translation)

            new_filename = f"{name}-t{ext}"
            new_path = os.path.join(translated_path, new_filename)  # Full path of the new file

            with open(new_path, "w", encoding="utf-8") as file:
                file.write(final_translation)

            print('-'*50)