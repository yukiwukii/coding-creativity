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

# Add argument parser
parser = argparse.ArgumentParser(description='Process model responses')
parser.add_argument('--type', type=str, required=True, help='Type parameter for processing')
parser.add_argument('--folder', type=str, required=True, help='Type folder for processing')
parser.add_argument('--port', type=str, required=True, help='Port')
args = parser.parse_args()

# Access the type argument with: args.type
print(f"Type parameter: {args.type}")
model_folder = "Llama70B"

# with open(f"Mistral7B/{args.folder}/CoveMistral7_part1.json",'r') as file:
#     dataneo=file.read()

# with open("datasets/CodeForce/inference/Llama1B/cove1/part3.json",'r') as file:
#     dataneo=json.load(file)

with open(f"datasets/CodeForce/inference/{model_folder}/{args.folder}/part1.json",'r') as file:
    dataneo=json.load(file)

    
local_llm_config = {
    "config_list": [
        {
            "model": "llama70-2:latest",  # CHANGE THIS!!!!
            "api_key": "ollama",  # 
            "base_url": f"http://localhost:{args.port}/v1",  # Your URL
            "price": [0, 0],  # Put in price per 1K tokens [prompt, response] as free!
           
        }
    ],
    "cache_seed": None,  # Turns off caching, useful for testing different models
    "timeout": 1500,
    "temperature": 1.0,
    "top_p": 1.0,
}


for idx,j in enumerate(dataneo):
    coveq=[]
    covea=[]
    
    if j.get("baseq"):
        print(f"{idx} is already processed. Skipping.")
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

        chat_results1 = user.initiate_chats(
        [
            {
                "recipient": sysv1,
                "message":f"""User: '{i}'\n\nAssistant: '{j["baseres"][idxinner]}'\n\nAs an AI assistant, please list verification questions to check the factual accuracy of the assistant's response code snippet above and whether it adheres to the user's instructions.""",
                "max_turns": 1,
                "summary_method":"last_msg",
                "clear_history": False
                
            },
            {
                "recipient": sysv1,
                "message":f"""Code Snippet: '{j["baseres"][idxinner]}'\n\nTo verify the code snippet above, answer each of these verification questions in the Context below.\n""",
                "max_turns": 1,
                "summary_method": "last_msg",
                "clear_history": False
                
            }])
        coveq.append(chat_results1[0].chat_history[1]["content"])
        covea.append(chat_results1[0].chat_history[3]["content"])

        print("Q is", chat_results1[0].chat_history[1]["content"])
        print('-' * 25)
            
    #         {
    #             "recipient": sysv1,
    #             "message": f"""User's Query: {prompts_data.loc[i,"FinalPrompt"]}\nInitial Response Story: {chatter[0].chat_history[1]["content"]}\nAbove is your initial response story to the user's query. Based on the verification results in the Context below, please rectify your story and provide a final, corrected story.""",
    #             "max_turns": 1,
    #             "summary_method": "last_msg",
    #             "clear_history": False
                
    #         },
    #     ]
    # )

    j["baseq"]=coveq
    j["basea"]=covea

    with open(f"datasets/CodeForce/inference/{model_folder}/{args.folder}/{args.type}.json", "w") as cove_file:
        json.dump(dataneo, cove_file, indent=4)
    print(f"Saved at {idx}.")

    # prompts_data.loc[i,"FinalQuestions"]=chat_results1[0].chat_history[1]["content"]
    # prompts_data.loc[i,"FinalAnswers"]=chat_results1[0].chat_history[3]["content"]


print(f"datasets/CodeForce/inference/{model_folder}/{args.folder}/{args.type}.json")
