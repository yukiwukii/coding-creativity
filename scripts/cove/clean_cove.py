import json
import re

with open('datasets/CodeForce/inference/Llamacode/cove3/part3.json','r') as file:
        dataneo=file.read()
dataneo=json.loads(dataneo)

for idx,j in enumerate(dataneo):
  for idx2,i in enumerate(j["outputs"]):
    matches = re.findall(r"```python(.*?)```", i, re.DOTALL)
    last_code_block = matches[-1] if matches else ""
    t3="```python"+last_code_block+"```"
    j["outputs"][idx2]=t3

with open("datasets/CodeForce/inference-mohor/GreedyNonGreedy/Neocoder/non-greedy/Llamacode/Llamacode_sample=199_dp=5_cove3.json", "w") as cove_file:
    json.dump(dataneo, cove_file, indent=4)