from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import json
import pandas as pd
import numpy as np
import torch
from accelerate.test_utils.testing import get_backend
from transformers import BitsAndBytesConfig
with open('Mistral7B/Base1.json','r') as file:
	data = json.load(file)
with open('Mistral7B/Base2.json','r') as file:
	data2 = json.load(file)
with open('Mistral7B/Base3.json', 'r') as file:
	data3 = json.load(file)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct",
# 		device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
		device_map="auto", quantization_config=quantization_config)

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
candidate_premature_layers = list(range(0, model.config.num_hidden_layers, 2))
set_seed(42)
for idx,i in enumerate(data):

	if i.get("outputs"):
		print(f"{idx} is already processed. Skipping.")
		continue

	# if idx <= 161:
	# 	print("We are skipping this index.")
	# 	continue

	base1=[]
	base2=[]
	base3=[]
	for j in i["problem_statements"]:
	
		chat = [
        {"role": "user", "content": j},
    	]

		prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
		inputs = tokenizer(prompt, return_tensors="pt").to(device)

		base1_output = model.generate(**inputs, do_sample=True, max_new_tokens=800)
		base2_output = model.generate(**inputs, do_sample=True, max_new_tokens=800)
		base3_output = model.generate(**inputs, do_sample=True, max_new_tokens=800)
		base1.append((tokenizer.batch_decode(base1_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True, pad_token_id=tokenizer.eos_token_id))[0])
		base2.append((tokenizer.batch_decode(base2_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True, pad_token_id=tokenizer.eos_token_id))[0])
		base3.append((tokenizer.batch_decode(base3_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True, pad_token_id=tokenizer.eos_token_id))[0])
	i["outputs"]=base1
	data2[idx]["outputs"]=base2
	data3[idx]["outputs"]=base3

	if idx % 5 == 0:
		with open("Mistral7B/Base1.json", "w") as dola_file:
			json.dump(data, dola_file, indent=4)

		with open("Mistral7B/Base2.json", "w") as base2_file:
			json.dump(data2, base2_file, indent=4)

		with open("Mistral7B/Base3.json", "w") as base3_file:
			json.dump(data3, base3_file, indent=4)
		print(f"Saved at index {idx}.")
	
with open("Mistral7B/Base1.json", "w") as dola_file:
    json.dump(data, dola_file, indent=4)

with open("Mistral7B/Base2.json", "w") as base2_file:
    json.dump(data2, base2_file, indent=4)

with open("Mistral7B/Base3.json", "w") as base3_file:
    json.dump(data3, base3_file, indent=4)
