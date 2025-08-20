import sys
import argparse
print(sys.prefix)
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()
import json
import pandas as pd
import numpy as np
from autogen import ConversableAgent, UserProxyAgent, get_config_list


model_folder = "gptoss"

with open('raginftemplate.json','r') as file:
    dataneo = json.load(file)

local_llm_config = {
    "config_list": [
        {
            "model": "gpt-oss:20b", #sjo/deepseek-r1-8b-llama-distill-abliterated-q8_0:latest
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
            "price": [0, 0],
        }
    ],
    "cache_seed": None,
    "timeout": 1500,
    "temperature": 0
}

def extract_after_think(response_text):
    """
    Extract content that appears after </think> tag.
    If no </think> tag is found, return the original response.
    """
    if "</think>" in response_text:
        # Split by </think> and take everything after it
        parts = response_text.split("</think>", 1)
        if len(parts) > 1:
            return parts[1].strip()
    
    # If no </think> tag found, return original response
    return response_text

user = ConversableAgent(
    name="User",
    system_message="",
    llm_config=local_llm_config,
    human_input_mode="NEVER",
)

sysv1 = ConversableAgent(
    name="Sys",
    system_message="You may only generate Python code. Do not provide any explanation in your answer.",
    llm_config=local_llm_config,
    human_input_mode="NEVER",
)

for idx, j in enumerate(dataneo):
    print(f"Index is {idx}")
    coveout = []    
    
    # if j.get("baseres"):
    #     print(f"{idx} is already processed. Skipping.")
    #     continue
    
    for i in j["full_queries"]:

        chatter = user.initiate_chats([
            {
                "recipient": sysv1,
                "message": i,
                "max_turns": 1,
                "summary_method": "last_msg",
                "clear_history": True
            }
        ])
        
        # Extract the raw response
        raw_response = chatter[0].chat_history[1]["content"]
        
        # Filter to get only content after </think>
        filtered_response = extract_after_think(raw_response)
        
        coveout.append(filtered_response)
    
    j["outputs"] = coveout
    
    with open(f"datasets/CodeForce/inference/{model_folder}/Rag1.json", "w") as cove_file:
        json.dump(dataneo, cove_file, indent=4)


print(f"Saved at datasets/CodeForce/inference/{model_folder}/Rag1.json")