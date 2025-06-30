from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import json
import pandas as pd
import numpy as np
import torch
from accelerate.test_utils.testing import get_backend
from transformers import BitsAndBytesConfig
with open('Mistral7B/dola1.json','r') as file:
	data = json.load(file)
with open('Mistral7B/dola2.json','r') as file:
	data2 = json.load(file)
with open('Mistral7B/dola3.json','r') as file:
	data3 = json.load(file)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
		device_map="auto", quantization_config=quantization_config)

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
candidate_premature_layers_low = list(range(2, model.config.num_hidden_layers, 2))
print(candidate_premature_layers_low)
candidate_premature_layers_medium = list(range(2,model.config.num_hidden_layers, 2))
print(candidate_premature_layers_medium)
candidate_premature_layers_high = list(range(2, model.config.num_hidden_layers, 2))
print(candidate_premature_layers_high)
set_seed(42)

for idx,i in enumerate(data):
	# if(idx<100):
	# 	continue
	if i.get("outputs"):
		print(f"{idx} is already processed. Skipping.")
		continue

	dola_low=[]
	dola_med=[]
	dola_high=[]
	for j in i["problem_statements"]:
		print("Working on index", idx)
	
		chat = [
        {"role": "user", "content": j},
		]

		prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
		inputs = tokenizer(prompt, return_tensors="pt").to(device)
		print(inputs)

		dola_low_output = model.generate(**inputs, do_sample=False,max_new_tokens=800, dola_layers=candidate_premature_layers_low, pad_token_id=tokenizer.eos_token_id)
		dola_med_output = model.generate(**inputs, do_sample=False,max_new_tokens=800,dola_layers=candidate_premature_layers_medium, pad_token_id=tokenizer.eos_token_id)
		dola_high_output = model.generate(**inputs, do_sample=False,max_new_tokens=800,dola_layers=candidate_premature_layers_high, pad_token_id=tokenizer.eos_token_id)
		dola_low.append((tokenizer.batch_decode(dola_low_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])
		dola_med.append((tokenizer.batch_decode(dola_med_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])
		dola_high.append((tokenizer.batch_decode(dola_high_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])
	i["outputs"]=dola_low
	data2[idx]["outputs"]=dola_med
	data3[idx]["outputs"]=dola_high

	if idx % 5 == 0:
		with open("Mistral7B/dola1.json", "w") as dola_file:
			json.dump(data, dola_file, indent=4)
		with open("Mistral7B/dola2.json", "w") as base_file:
			json.dump(data2, base_file, indent=4)
		with open("Mistral7B/dola3.json", "w") as base_file:
			json.dump(data3, base_file, indent=4)
		print(f"Saved after processing {idx + 1} entries.")
	 
with open("Mistral7B/dola1.json", "w") as dola_file:
    json.dump(data, dola_file, indent=4)
with open("Mistral7B/dola2.json", "w") as base_file:
    json.dump(data2, base_file, indent=4)
with open("Mistral7B/dola3.json", "w") as base_file:
    json.dump(data3, base_file, indent=4)