import json
import re

with open('datasets/CodeForce/inference-mohor/GreedyNonGreedy/Neocoder/non-greedy/Llama1B/Llama-1B-Instruct_sample=199_dp=5_rag3.json','r') as file:
        dataneo=file.read()
dataneo=json.loads(dataneo)

for idx,j in enumerate(dataneo):
  for idx2,i in enumerate(j["outputs"]):
    matches = re.findall(r"```python(.*?)```", i, re.DOTALL)
    last_code_block = matches[-1] if matches else ""
    t3="```python"+last_code_block+"```"
    j["outputs"][idx2]=t3

with open("datasets/CodeForce/inference-mohor/GreedyNonGreedy/Neocoder/non-greedy/Llama1B/Llama-1B-Instruct_sample=199_dp=5_rag3-cleaned.json", "w") as cove_file:
    json.dump(dataneo, cove_file, indent=4)