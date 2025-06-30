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
from autogen import ConversableAgent, UserProxyAgent

# Add argument parser
parser = argparse.ArgumentParser(description='Process model responses')
parser.add_argument('--folder', type=str, required=True, help='Type folder')
parser.add_argument('--type', type=str, required=True, help='Type parameter for processing')
args = parser.parse_args()

# Access the type argument with: args.type
print(f"Type parameter: {args.type}")

# with open('Llama-3.1-8B-Instruct_sample=199_dp=5.json','r') as file:
#     dataneo = file.read()
# dataneo = json.loads(dataneo)

with open('Llama70B/cove2/CoveLlama3_1_part3.json', 'r') as file:
    dataneo = json.load(file)

local_llm_config = {
    "config_list": [
        {
            "model": "llama3.1:70b",
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
            "price": [0, 0],
        }
    ],
    "cache_seed": None,
}

from autogen import AssistantAgent, UserProxyAgent

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

for idx, j in enumerate(dataneo):
    coveout = []    
    
    if j.get("baseres"):
        print(f"{idx} is already processed. Skipping.")
        continue
    
    for i in j["problem_statements"]:
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
        
        chatter = user.initiate_chats([
            {
                "recipient": sysv1,
                "message": i,
                "max_turns": 1,
                "summary_method": "last_msg",
                "clear_history": False
            }
        ])
        
        # Extract the raw response
        raw_response = chatter[0].chat_history[1]["content"]
        
        # Filter to get only content after </think>
        filtered_response = extract_after_think(raw_response)
        
        coveout.append(filtered_response)
    
    j["baseres"] = coveout
    
    # Save every 5 iterations - you can use args.type in the filename if needed
    if idx % 5 == 0:
        with open(f"Llama70B/{args.folder}/CoveLlama3_1_{args.type}.json", "w") as cove_file:
            json.dump(dataneo, cove_file, indent=4)

# Final save
with open(f"Llama70B/{args.folder}/CoveLlama3_1_{args.type}.json", "w") as cove_file:
    json.dump(dataneo, cove_file, indent=4)

print(f"Saved at Llama70B/{args.folder}/CoveLlama3_1_{args.type}.json")