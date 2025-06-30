import json

# load your JSON
with open('datasets/CodeForce/llama8_summary.json', 'r') as f:
    data = json.load(f)

# for each model, convert all of its numeric‐array fields from %→fraction
for model in data.values():
    for field, values in model.items():
        # only convert lists of numbers
        if isinstance(values, list) and all(isinstance(x, (int, float)) for x in values):
            model[field] = [x / 100 for x in values]

# write it back (or to a new file)
with open('datasets/CodeForce/neoresults/Llama8B/summary.json', 'w') as f:
    json.dump(data, f, indent=2)