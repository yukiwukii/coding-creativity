import json
import re

# Load the JSON file
with open('datasets/CodeForce/inference/gptoss/Base1.json','r') as file:
    dataneo = file.read()
dataneo = json.loads(dataneo)

# Process each item in the dataset
for idx, j in enumerate(dataneo):
    for idx2, i in enumerate(j["outputs"]):
        # First try to find code blocks in markdown format
        matches = re.findall(r"```python(.*?)```", i, re.DOTALL)
        
        if matches:
            # If we found markdown code blocks, use the last one
            last_code_block = matches[-1]
            t3 = "```python" + last_code_block + "```"
        else:
            # If no markdown blocks found, look for code after "assistantfinal"
            assistantfinal_match = re.search(r"assistantfinal\s*(.*)", i, re.DOTALL)
            if assistantfinal_match:
                code_after_assistantfinal = assistantfinal_match.group(1).strip()
                # Wrap the extracted code in markdown format
                t3 = "```python\n" + code_after_assistantfinal + "\n```"
            else:
                # If neither pattern found, keep original
                t3 = i
        
        j["outputs"][idx2] = t3

# Save the cleaned data
with open("datasets/CodeForce/inference/gptoss/Base1_cleaned.json", "w") as cove_file:
    json.dump(dataneo, cove_file, indent=4)