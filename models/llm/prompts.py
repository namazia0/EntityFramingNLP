from dotenv import load_dotenv
import requests
import json
import yaml
import os

load_dotenv(dotenv_path="config/config.env")

class Prompts:
    def __init__(self, 
                tabby_api_url = "http://localhost:5001/v1",
                tabby_api_key = os.getenv("TABBY_API_KEY")):

        self.tabby_api_url = tabby_api_url
        self.tabby_api_key = tabby_api_key
    
    def tabby_api(self, prompt, max_tokens):
        with open("../tabbyAPI/config.yml", "r") as file:
            config = yaml.safe_load(file)
        llm_model = config["model"]["model_name"]
        # print("llm model: ", llm_model)

        if not self.tabby_api_key:
            raise ValueError("API key is not set in the environment variable TABBY_API_KEY.")

        url = f'{self.tabby_api_url}/chat/completions'
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {self.tabby_api_key}'
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
    
    def create_context(self, title, document, chunk, entity):      
        # The following prompt is extracted from https://www.anthropic.com/news/contextual-retrieval  
        prompt = f"""Please give a short succinct context to situate the chunk and the namend entity within the overall document for the purposes of improving entity role classification. Answer only with the succinct context and nothing else. 
        
        Title: {title}
        Document: {document}

        Chunk: {chunk}
        Named entity: {entity}

        Do not preamble.
        """
        max_tokens = 100
        answer = self.tabby_api(prompt, max_tokens)
        return answer
    
    def generate_prediction_doc(self, title, article, entity, start_offset, end_offset):
        roles = 'Protagonist, Antagonist, Innocent'
        protagonist = '[Guardian, Martyr, Peacemaker, Rebel, Underdog, Virtuous]'
        antagonist = '[Instigator, Conspirator, Tyrant, Foreign Adversary, Traitor, Spy, Saboteur, Corrupt, Incompetent, Terrorist, Deceiver, Bigot]'
        innocents = '[Forgotten, Exploited, Victim, Scapegoat]'
        role_item = {}
        role_item['roles'] = roles
        role_item['protagonist'] = protagonist
        role_item['antagonist'] = antagonist
        role_item['innocent'] = innocents
        
        prompt = f"""
            ### TASK DESCRIPTION
            You are an AI assistant tasked with a narrative role classification problem. 
            Your goal is to assign roles that defines the Named Entity based on the Article and its Title.
            Start Offset and End Offset is the span where the Named Entity appears in the Article.
            The roles MUST precisely reflect the entity's characteristics.

            ### INPUT
            Title: {title}
            Article: {article}
            Named Entity: {entity}
            Start Offset: {start_offset}
            End Offset: {end_offset}

            ### INSTRUCTION
            Determine EXACTLY ONE main role: Protagonist, Antagonist, or Innocent

            ### AUTHORIZED SUB-ROLE LISTS ###
            You are a strict classifier. Do not generate any output outside of the given lists. DO NOT invent new sub-roles that are NOT listed. 
            You MUST select from the following list. If you generate a sub-role that is not in the list, you are making an ERROR.
            **IMPORTANT: Choose ONLY from the provided list.**  
            **Select STRICTLY 1-2 sub-roles that describes the named entity**:  
            - **If Protagonist, choose ONLY from:** {role_item['protagonist']}  
            - **If Antagonist, choose ONLY from:** {role_item['antagonist']}  
            - **If Innocent, choose ONLY from:** {role_item['innocent']} 
            Do not return None.

            ### OUTPUT FORMAT
            Please provide your answer STRICTLY in the following format: Main role: sub-role 1, sub-role 2
            Example:
            Antagonist: Tyrant, Spy 

            Do not preamble.

            Answer:
        """
        max_tokens = 15
        answer = self.tabby_api(prompt, max_tokens)
        return answer
    
    def generate_prediction(self, article_sentence, context, entity, language):
        """
        article_sentence: Give the LLM only the sentence where the entity lies in the article text. 
        contxt: context of the given sentence within the article
        entity: entity to predict its roles and sub-roles
        language: EN (english) or PT (portuguese)
        """

        prompt = ""
        # which in-context learning strategy you want to apply. 
        in_context_learning = int(os.getenv("IN_CONTEXT_LEARNING"))     # 0: zero-shot, 1: one-shot, 2: few-shot, 3: Chain-of-Thought (CoT)
        _context = int(os.getenv("CONTEXT"))     # 0: no additional context (only the extracted sentence), 1: with generated context
        
        # pre-defined roles and sub-roles
        roles = 'Protagonist, Antagonist, Innocent'
        protagonist = '[Guardian, Martyr, Peacemaker, Rebel, Underdog, Virtuous]'
        antagonist = '[Instigator, Conspirator, Tyrant, Foreign Adversary, Traitor, Spy, Saboteur, Corrupt, Incompetent, Terrorist, Deceiver, Bigot]'
        innocents = '[Forgotten, Exploited, Victim, Scapegoat]'
        role_item = {}
        role_item['roles'] = roles
        role_item['protagonist'] = protagonist
        role_item['antagonist'] = antagonist
        role_item['innocent'] = innocents

        if in_context_learning == 0:   # Zero-shot
            if _context == 1:    # with additional context
                prompt = f"""
                    ### TASK DESCRIPTION
                    You are an AI assistant tasked with a narrative role classification problem. 
                    Your goal is to assign roles that defines the Named Entity based on the Sentence and its Context.
                    The roles MUST precisely reflect the entity's characteristics.

                    ### INPUT
                    Sentence: {article_sentence}
                    Context: {context}
                    Named Entity: {entity}

                    ### INSTRUCTION
                    Determine EXACTLY ONE main role: Protagonist, Antagonist, or Innocent

                    ### AUTHORIZED SUB-ROLE LISTS ###
                    You are a strict classifier. Do not generate any output outside of the given lists. DO NOT invent new sub-roles that are NOT listed. 
                    You MUST select from the following list. If you generate a sub-role that is not in the list, you are making an ERROR.
                    **IMPORTANT: Choose ONLY from the provided list.**  
                    **Select STRICTLY 1-2 sub-roles that describes the named entity**:  
                    - **If Protagonist, choose ONLY from:** {role_item['protagonist']}  
                    - **If Antagonist, choose ONLY from:** {role_item['antagonist']}  
                    - **If Innocent, choose ONLY from:** {role_item['innocent']} 
                    Do not return None.

                    ### OUTPUT FORMAT
                    Please provide your answer STRICTLY in the following format: Main role: sub-role 1, sub-role 2
                    Example:
                    Antagonist: Tyrant, Spy 

                    Do not preamble.

                    Answer:
                """
            else:   # without additional context
                prompt = f"""
                    ### TASK DESCRIPTION
                    You are an AI assistant tasked with a narrative role classification problem. 
                    Your goal is to assign roles that defines the named entity based on the sentence.
                    The roles MUST precisely reflect the entity's characteristics.

                    ### INPUT
                    Sentence: {article_sentence}
                    Named Entity: {entity}

                    ### INSTRUCTION
                    Determine EXACTLY ONE main role: Protagonist, Antagonist, or Innocent

                    ### AUTHORIZED SUB-ROLE LISTS ###
                    You are a strict classifier. Do not generate any output outside of the given lists. DO NOT invent new sub-roles that are NOT listed. 
                    You MUST select from the following list. If you generate a sub-role that is not in the list, you are making an ERROR.
                    **IMPORTANT: Choose ONLY from the provided list.**  
                    **Select STRICTLY 1-2 sub-roles that describes the named entity**:  
                    - **If Protagonist, choose ONLY from:** {role_item['protagonist']}  
                    - **If Antagonist, choose ONLY from:** {role_item['antagonist']}  
                    - **If Innocent, choose ONLY from:** {role_item['innocent']} 
                    Do not return None.

                    ### OUTPUT FORMAT
                    Please provide your answer STRICTLY in the following format: Main role: sub-role 1, sub-role 2
                    Example:
                    Antagonist: Tyrant, Spy 

                    Do not preamble.

                    Answer:
                """

        if in_context_learning == 1: # One-shot
            if language == "EN":
                example_sentence = "The hero led the charge against the invaders."
                example_context = "The invaders were attempting to conquer the land, but the hero organized the resistance."
                example_entity = "Hero"
                example_answer = "Protagonist: Peacemaker, Guardian"
            elif language == "PT":
                example_sentence = "O herói liderou o ataque contra os invasores."
                example_context = "Os invasores estavam tentando conquistar a terra, mas o herói organizou a resistência."
                example_entity = "Herói"
                example_answer = "Protagonist: Peacemaker, Guardian"

            if _context == 1:
                prompt = f"""
                    ### TASK DESCRIPTION
                    You are an AI assistant tasked with a narrative role classification problem. 
                    Your goal is to assign roles that defines the Named Entity based on the Sentence and its Context.
                    The roles MUST precisely reflect the entity's characteristics.
                    
                    ### EXAMPLE
                    Sentence: {example_sentence}
                    Context: {example_context}
                    Named Entity: {example_entity}
                    Answer: {example_answer}

                    ### Now, for the given input:
                    Sentence: {article_sentence}
                    Context: {context}
                    Named Entity: {entity}

                    ### INSTRUCTION
                    Determine EXACTLY ONE main role: Protagonist, Antagonist, or Innocent

                    ### AUTHORIZED SUB-ROLE LISTS ###
                    You are a strict classifier. Do not generate any output outside of the given lists. DO NOT invent new sub-roles that are NOT listed. 
                    You MUST select from the following list. If you generate a sub-role that is not in the list, you are making an ERROR.
                    **IMPORTANT: Choose ONLY from the provided list.**  
                    **Select STRICTLY 1-2 sub-roles that describes the named entity**:  
                    - **If Protagonist, choose ONLY from:** {role_item['protagonist']}  
                    - **If Antagonist, choose ONLY from:** {role_item['antagonist']}  
                    - **If Innocent, choose ONLY from:** {role_item['innocent']} 
                    Do not return None.

                    ### OUTPUT FORMAT
                    Please provide your answer STRICTLY in the following format: Main role: sub-role 1, sub-role 2
                    Example:
                    Antagonist: Tyrant, Spy 

                    Do not preamble.

                    Answer:
                """
            else: 
                prompt = f"""
                    ### TASK DESCRIPTION
                    You are an AI assistant tasked with a narrative role classification problem. 
                    Your goal is to assign roles that defines the Named Entity based on the Sentence and its Context.
                    The roles MUST precisely reflect the entity's characteristics.
                    
                    ### EXAMPLE
                    Sentence: {example_sentence}
                    Named Entity: {example_entity}
                    Answer: {example_answer}

                    ### Now, for the given input:
                    Sentence: {article_sentence}
                    Named Entity: {entity}

                    ### INSTRUCTION
                    Determine EXACTLY ONE main role: Protagonist, Antagonist, or Innocent

                    ### AUTHORIZED SUB-ROLE LISTS ###
                    You are a strict classifier. Do not generate any output outside of the given lists. DO NOT invent new sub-roles that are NOT listed. 
                    You MUST select from the following list. If you generate a sub-role that is not in the list, you are making an ERROR.
                    **IMPORTANT: Choose ONLY from the provided list.**  
                    **Select STRICTLY 1-2 sub-roles that describes the named entity**:  
                    - **If Protagonist, choose ONLY from:** {role_item['protagonist']}  
                    - **If Antagonist, choose ONLY from:** {role_item['antagonist']}  
                    - **If Innocent, choose ONLY from:** {role_item['innocent']} 
                    Do not return None.

                    ### OUTPUT FORMAT
                    Please provide your answer STRICTLY in the following format: Main role: sub-role 1, sub-role 2
                    Example:
                    Antagonist: Tyrant, Spy 

                    Do not preamble.

                    Answer:
                """

        if in_context_learning == 2:  # Few-shot
            if language == "EN":
                examples = """
                    Sentence: The hero led the charge against the invaders.
                    Context: The invaders were attempting to conquer the land, but the hero organized the resistance.
                    Named Entity: Hero
                    Answer: Protagonist: Peacemaker, Guardian

                    Sentence: The villain schemed to overthrow the kingdom.
                    Context: The kingdom had been at peace for years, but the villain plotted to seize the throne through deception.
                    Named Entity: Villain
                    Answer: Antagonist: Tyrant

                    Sentence: The villagers were caught in the crossfire of the warring armies.
                    Context: The warring factions showed no concern for the lives of the villagers, leaving them defenseless and suffering.
                    Named Entity: Villagers
                    Answer: Innocent: Victim, Exploited

                    Sentence: The laborers were blamed for the factory fire despite having no involvement.
                    Context: The real cause of the fire was negligence by the management, but the laborers became scapegoats to protect those in power.
                    Named Entity: Laborers
                    Answer: Innocent: Scapegoat, Forgotten

                    Sentence: The corrupt official embezzled public funds for personal gain.
                    Context: Despite public outrage, the official used their position of power to exploit the system.
                    Named Entity: Corrupt Official
                    Answer: Antagonist: Corrupt, Deceiver
                """
                
            elif language == "PT":
                examples = """
                    Sentence: O herói liderou o ataque contra os invasores.
                    Context: Os invasores estavam tentando conquistar a terra, mas o herói organizou a resistência.
                    Named Entity: Herói
                    Answer: Protagonist: Peacemaker, Guardian

                    Sentence: O vilão tramou para derrubar o reino.
                    Context: O reino estava em paz há anos, mas o vilão planejou tomar o trono por meio de engano.  
                    Named Entity: Vilão
                    Answer: Antagonist: Tyrant

                    Sentence: Os aldeões foram pegos no fogo cruzado dos exércitos em guerra.  
                    Context: As facções em guerra não demonstraram preocupação com as vidas dos aldeões, deixando-os indefesos e sofrendo.
                    Named Entity: Aldeões
                    Answer: Innocent: Victim, Exploited
                """
                # Sentence: Os trabalhadores foram culpados pelo incêndio na fábrica, apesar de não terem envolvimento.
                # Context: A verdadeira causa do incêndio foi a negligência da administração, mas os trabalhadores se tornaram bodes expiatórios para proteger os poderosos.
                # Named Entity: Trabalhadores
                # Answer: Innocent: Scapegoat, Forgotten

                # Sentence: O oficial corrupto desviou fundos públicos para benefício próprio.
                # Context: Apesar da indignação pública, o oficial usou sua posição de poder para explorar o sistema.
                # Named Entity: Corrupto
                # Answer: Antagonist: Corrupt, Deceiver

            prompt = f"""
                ### TASK DESCRIPTION
                You are an AI assistant tasked with a narrative role classification problem. 
                Your goal is to assign roles that defines the named entity based on the sentence and its context.
                The roles MUST precisely reflect the entity's characteristics.
                
                ### EXAMPLES
                {examples}

                ### Now, for the given input:
                Sentence: {article_sentence}
                Context: {context}
                Named Entity: {entity}

                ### INSTRUCTION
                Determine EXACTLY ONE main role: Protagonist, Antagonist, or Innocent

                ### AUTHORIZED SUB-ROLE LISTS ###
                You are a strict classifier. Do not generate any output outside of the given lists. DO NOT invent new sub-roles that are NOT listed. 
                You MUST select from the following list. If you generate a sub-role that is not in the list, you are making an ERROR.
                **IMPORTANT: Choose ONLY from the provided list.**  
                **Select STRICTLY 1-2 sub-roles that describes the named entity**:  
                - **If Protagonist, choose ONLY from:** {role_item['protagonist']}  
                - **If Antagonist, choose ONLY from:** {role_item['antagonist']}  
                - **If Innocent, choose ONLY from:** {role_item['innocent']} 
                Do not return None.

                ### OUTPUT FORMAT
                Please provide your answer STRICTLY in the following format: Main role: sub-role 1, sub-role 2
                Example:
                Antagonist: Tyrant, Spy 

                Do not preamble.

                Answer:
            """
        
        if in_context_learning == 3:     # Chain-of-Thought
            prompt = f"""
                ### TASK DESCRIPTION
                You are an AI assistant tasked with a narrative role classification problem. 
                Your goal is to assign roles that defines the named entity based on the sentence and its context.
                The roles MUST precisely reflect the entity's characteristics.

                ### INPUT
                Sentence: {article_sentence}
                Context: {context}
                Named Entity: {entity}

                ### INSTRUCTION
                Determine EXACTLY ONE main role: Protagonist, Antagonist, or Innocent

                ### AUTHORIZED SUB-ROLE LISTS ###
                You are a strict classifier. Do not generate any output outside of the given lists. DO NOT invent new sub-roles that are NOT listed. 
                You MUST select from the following list. If you generate a sub-role that is not in the list, you are making an ERROR.
                **IMPORTANT: Choose ONLY from the provided list.**  
                **Select STRICTLY 1-2 sub-roles that describes the named entity**:  
                - **If Protagonist, choose ONLY from:** {role_item['protagonist']}  
                - **If Antagonist, choose ONLY from:** {role_item['antagonist']}  
                - **If Innocent, choose ONLY from:** {role_item['innocent']} 
                Do not return None.

                Explain your choice for the main role and sub roles step by step. Think step by step.

                Please give your final answer at the end STRICTLY in the following format: Main role: sub-role 1, sub-role 2
                Example:
                Antagonist: Tyrant, Spy 
            """

        if in_context_learning == 3:
            max_tokens = 300
        else:
            max_tokens = 15
        answer = self.tabby_api(prompt, max_tokens)
        return answer