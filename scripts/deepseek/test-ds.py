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

local_llm_config = {
    "config_list": [
        {
            "model": "cove-deepseek-config",
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
            "price": [0, 0],
        }
    ],
    "cache_seed": None,
}

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
        "message": '''You are a Python code generator, only return the import and python function. Input will be an very detailed description of task, output will be the code.
The input will be from command line, and the output will be printed to the console as well. Your result will be solely a function named solve(), and do not call this function in your code.
Make sure the code is free of bug and can pass the test cases provided. You can use any library you want. The test cases are provided in the code. Do not call the solve() function in your code. 
Programming Problem: A. Line Trip
There is a road, which can be represented as a number line. You are located in the point $$$0$$$ of the number line, and you want to travel from the point $$$0$$$ to the point $$$x$$$, and back to the point $$$0$$$.
You travel by car, which spends $$$1$$$ liter of gasoline per $$$1$$$ unit of distance travelled. When you start at the point $$$0$$$, your car is fully fueled (its gas tank contains the maximum possible amount of fuel).
There are $$$n$$$ gas stations, located in points $$$a_1, a_2, \dots, a_n$$$. When you arrive at a gas station, you fully refuel your car.
Note that you can refuel only at gas stations, and there are no gas stations in points $$$0$$$ and $$$x$$$
.
You have to calculate the minimum possible volume of the gas tank in your car (in liters) that will allow you to travel from the point $$$0$$$ to the point $$$x$$$ and back to the point $$$0$$$.
Input
The first line contains one integer $$$t$$$ ($$$1 \le t \le 1000$$$) — the number of test cases.
Each test case consists of two lines:
the first line contains two integers $$$n$$$ and $$$x$$$ ($$$1 \le n \le 50$$$; $$$2 \le x \le 100$$$);
the second line contains $$$n$$$ integers $$$a_1, a_2, \dots, a_n$$$ ($$$0 < a_1 < a_2 < \dots < a_n < x$$$).
Output
For each test case, print one integer — the minimum possible volume of the gas tank in your car that will allow you to travel from the point $$$0$$$ to the point $$$x$$$ and back.
Example
Input
3
3 7
1 2 5
3 6
1 2 5
1 10
7
Output
4
3
7
Note
In the first test case of the example, if the car has a gas tank of $$$4$$$ liters, you can travel to $$$x$$$ and back as follows:
travel to the point $$$1$$$, then your car's gas tank contains $$$3$$$ liters of fuel;
refuel at the point $$$1$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$2$$$, then your car's gas tank contains $$$3$$$ liters of fuel;
refuel at the point $$$2$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$5$$$, then your car's gas tank contains $$$1$$$ liter of fuel;
refuel at the point $$$5$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$7$$$, then your car's gas tank contains $$$2$$$ liters of fuel;
travel to the point $$$5$$$, then your car's gas tank contains $$$0$$$ liters of fuel;
refuel at the point $$$5$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$2$$$, then your car's gas tank contains $$$1$$$ liter of fuel;
refuel at the point $$$2$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$1$$$, then your car's gas tank contains $$$3$$$ liters of fuel;
refuel at the point $$$1$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$0$$$, then your car's gas tank contains $$$3$$$ liters of fuel.''',
        "max_turns": 1,
        "summary_method": "last_msg",
        "clear_history": False
    }
])

raw_response = chatter[0].chat_history[1]["content"]
print('-' * 50)
print(extract_after_think(raw_response))