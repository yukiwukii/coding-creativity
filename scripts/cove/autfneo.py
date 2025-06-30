import sys
print(sys.prefix)
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()
import json
import pandas as pd
import numpy as np
from autogen import ConversableAgent, UserProxyAgent
import argparse

parser = argparse.ArgumentParser(description='Process model responses')
parser.add_argument('--type', type=str, required=True, help='Type parameter for processing')
parser.add_argument('--folder', type=str, required=True, help='Type folder for processing')
args = parser.parse_args()

# with open(f'Llama70B/{args.folder}/CoveLlama3_1_part2.json','r') as file:
#     dataneo=file.read()
# dataneo=json.loads(dataneo)

with open("Llama70B/cove2/CoveLlama3_1_part3.json",'r') as file:
    dataneo=json.load(file)

local_llm_config = {
    "config_list": [
        {
            "model": "llama3.1:70b", 
            "api_key": "ollama", 
            "base_url": "http://localhost:11434/v1", 
            "price": [0, 0],  # Put in price per 1K tokens [prompt, response] as free!
           
        }
    ],
    "cache_seed": None,  # Turns off caching, useful for testing different models
}

from autogen import AssistantAgent, UserProxyAgent
for idx,j in enumerate(dataneo):
    coveo=[]
    
    if j.get("outputs"):
        print(f'{idx} is already processed. Skipping.')
        continue
    
    for idxinner,i in enumerate(j["problem_statements"]):
        user = ConversableAgent(
        name="User",
        system_message="",
        llm_config=local_llm_config,
        human_input_mode="NEVER",
        
    )
        sysv1 = ConversableAgent(
        name="Sys",
        system_message="",
        llm_config=local_llm_config,
        human_input_mode="NEVER",
    
    )

        
    #   chatter=user.initiate_chats(    [
    #       {
    #           "recipient": sysv1,
    #           "message": prompts_data.loc[i,"FinalPrompt"],
    #           "max_turns": 1,
    #           "summary_method": "last_msg",
    #           "clear_history": False           
    #        }])

    #  chat_results1 = user.initiate_chats(
    #   [
    #       {
    #           "recipient": sysv1,
    #           "message":f"""User: '{prompts_data.loc[i,"FinalPrompt"]}'\n\nAssistant: '{prompts_data.loc[i,"FinalBaseStory"]}'\n\nAs an AI assistant, please list verification questions to check the factual accuracy of the assistant's response story above and whether it adheres to the user's instructions.""",
    #           "max_turns": 1,
    #           "summary_method":"last_msg",
    #           "clear_history": False
    #          
    #       },
    #       {
    #           "recipient": sysv1,
    #           "message":f"""Story: '{prompts_data.loc[i,"FinalBaseStory"]}'\n\nTo verify the story above, answer each of these verification questions in the Context below.\n""",
    #           "max_turns": 1,
    #           "summary_method": "last_msg",
    #           "clear_history": False
    #           
    #         }])

        chat_results1 = user.initiate_chats(
        [
            {
                "recipient": sysv1,
                "message": f"""User: '{i}'\n\nAssistant (Initial Response Code Snippet): '{j["baseres"][idxinner]}'\n\nVerification Questions: '{j["baseq"][idxinner]}'\n\nVerification Answers: '{j["basea"][idxinner]}'\n\nBased on the verification results, please provide a final, corrected code snippet to the user's query.""",
                "max_turns": 1,
                "summary_method": "last_msg",
                "clear_history": False
                
            },
        ])

        coveo.append(chat_results1[0].chat_history[1]["content"])
    j["outputs"]=coveo
    # prompts_data.loc[i,"FinalResponse"]=chat_results1[0].chat_history[1]["content"]
   # prompts_data.loc[i,"FinalAnswers"]=chat_results1[0].chat_history[3]["content"]
    if idx % 5 == 0:
        with open(f"Llama70B/{args.folder}/CoveLlama3_1_{args.type}.json", "w") as cove_file:
            json.dump(dataneo, cove_file, indent=4)
        print(f"Saved at {idx}.")
    
with open(f"Llama70B/{args.folder}/CoveLlama3_1_{args.type}.json", "w") as cove_file:
    json.dump(dataneo, cove_file, indent=4)

print(f"Saved at Llama70B/{args.folder}/CoveLlama3_1_{args.type}.json")