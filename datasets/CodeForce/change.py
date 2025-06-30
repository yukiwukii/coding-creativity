import json

def convert_fractions_to_percentages(data):
    """
    Recursively convert all numerical values in a data structure from fractions to percentages.
    """
    if isinstance(data, dict):
        return {key: convert_fractions_to_percentages(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_fractions_to_percentages(item) for item in data]
    elif isinstance(data, (int, float)):
        return data * 100
    else:
        return data

# Read the JSON file
with open('datasets/CodeForce/llama70_summary.json', 'r') as file:
    data = json.load(file)

# Convert fractions to percentages
converted_data = convert_fractions_to_percentages(data)

# Write the converted data to a new file
with open('datasets/CodeForce/llama70_summary_percentages.json', 'w') as file:
    json.dump(converted_data, file, indent=2)

print("Conversion complete! The converted data has been saved to 'llama70_summary_percentages.json'")

# Optional: Print a sample of the converted data to verify
print("\nSample of converted data:")
first_key = list(converted_data.keys())[0]
print(f"{first_key}:")
print(json.dumps(converted_data[first_key], indent=2))