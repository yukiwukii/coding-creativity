from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import json
import pandas as pd
import numpy as np
import torch
from accelerate.test_utils.testing import get_backend
from transformers import BitsAndBytesConfig

model_folder = "Llama8B"

with open(f'Llama-3.1-8B-Instruct_sample=199_dp=5.json','r') as file:
	data = json.load(file)
with open(f'Llama-3.1-8B-Instruct_sample=199_dp=5.json','r') as file:
	data2 = json.load(file)
with open(f'Llama-3.1-8B-Instruct_sample=199_dp=5.json','r') as file:
	data3 = json.load(file)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
		device_map="auto", quantization_config=quantization_config)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct",
# 		device_map="auto", quantization_config=quantization_config)

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
# 		device_map="auto", quantization_config=quantization_config)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",
# 		device_map="auto")

# tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
# model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b",
# 		device_map="auto")

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
all_buckets = model.config.num_hidden_layers//3
print(f"We are contrasting with the first {all_buckets} layers.")
candidate_premature_layers = list(range(all_buckets * 2, model.config.num_hidden_layers, 2))
# set_seed(42)

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
		dola_low_output = model.generate(**inputs, do_sample=True, max_new_tokens=800, dola_layers=candidate_premature_layers, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, num_beam_groups=1, num_beams=1, temperature=1.0)
		dola_med_output = model.generate(**inputs, do_sample=True, max_new_tokens=800, dola_layers=candidate_premature_layers, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, num_beam_groups=1, num_beams=1, temperature=1.0)
		dola_high_output = model.generate(**inputs, do_sample=True, max_new_tokens=800, dola_layers=candidate_premature_layers, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, num_beam_groups=1, num_beams=1, temperature=1.0)
		dola_low.append((tokenizer.batch_decode(dola_low_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])
		dola_med.append((tokenizer.batch_decode(dola_med_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])
		dola_high.append((tokenizer.batch_decode(dola_high_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0])
	i["outputs"]=dola_low
	data2[idx]["outputs"]=dola_med
	data3[idx]["outputs"]=dola_high

	with open(f"datasets/CodeForce/inference/{model_folder}/dola1-low.json", "w") as dola_file:
		json.dump(data, dola_file, indent=4)
	with open(f"datasets/CodeForce/inference/{model_folder}/dola2-low.json", "w") as base_file:
		json.dump(data2, base_file, indent=4)
	with open(f"datasets/CodeForce/inference/{model_folder}/dola3-low.json", "w") as base_file:
		json.dump(data3, base_file, indent=4)
	print(f"Saved at index {idx}.")
	 