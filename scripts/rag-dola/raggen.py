from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import json
import pandas as pd
import numpy as np
import torch
from accelerate.test_utils.testing import get_backend
from transformers import BitsAndBytesConfig
with open('Mistral7B/Rag1.json','r') as file:
	data=file.read()
with open('Mistral7B/Rag2.json','r') as file:
	data2=file.read()
with open('Mistral7B/Rag3.json','r') as file:
	data3=file.read()
data=json.loads(data)
data2=json.loads(data2)
data3=json.loads(data3)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
		device_map="auto", quantization_config=quantization_config)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct",
# 		device_map="auto", quantization_config=quantization_config)

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
candidate_premature_layers = list(range(0, model.config.num_hidden_layers, 2))
set_seed(42)
for idx,i in enumerate(data):
	if(idx<146):
		continue
	# if(idx==100):
	# 	break
	out1=[]
	out2=[]
	out3=[]
	for j in i["full_queries"]:
	
		chat = [
        {"role": "user", "content": j},
    ]

		prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
		inputs = tokenizer(prompt, return_tensors="pt").to(device)

		output1 = model.generate(**inputs, do_sample=False,max_new_tokens=800)
		output2 = model.generate(**inputs, do_sample=False,max_new_tokens=800)
		output3 = model.generate(**inputs, do_sample=False,max_new_tokens=800)
		out1.append((tokenizer.batch_decode(output1[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])
		out2.append((tokenizer.batch_decode(output2[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])
		out3.append((tokenizer.batch_decode(output3[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])

	i["outputs"]=out1
	data2[idx]["outputs"]=out2
	data3[idx]["outputs"]=out3

	if idx % 5 == 0:
		with open("Mistral7B/Rag1.json", "w") as dola_file:
			json.dump(data, dola_file, indent=4)

		with open("Mistral7B/Rag2.json", "w") as base_file:
			json.dump(data2, base_file, indent=4)

		with open("Mistral7B/Rag3.json", "w") as base_file:
			json.dump(data3, base_file, indent=4)
		print(f"Saved at index {idx}.")
			
with open("Mistral7B/Rag1.json", "w") as dola_file:
    json.dump(data, dola_file, indent=4)

with open("Mistral7B/Rag2.json", "w") as base_file:
    json.dump(data2, base_file, indent=4)

with open("Mistral7B/Rag3.json", "w") as base_file:
    json.dump(data3, base_file, indent=4)