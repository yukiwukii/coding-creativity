import json

def remove_techniques_from_json(input_file, output_file=None):
    """
    Remove all 'techniques' entries from a JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file (optional)
    """
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Remove 'techniques' key from each problem entry
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'techniques' in item:
                del item['techniques']
    elif isinstance(data, dict):
        # If the root is a dict, check if it has techniques or if it contains problems
        if 'techniques' in data:
            del data['techniques']
        # Check for nested structures that might contain problems
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and 'techniques' in item:
                        del item['techniques']
    
    # Write to output file
    if output_file is None:
        output_file = input_file
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully removed 'techniques' entries from {input_file}")
    print(f"Output saved to: {output_file}")

# Usage example
if __name__ == "__main__":
    # Replace 'your_file.json' with your actual JSON file path
    input_file = input("Enter the path to your JSON file: ").strip()
    
    try:
        remove_techniques_from_json(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format. {e}")
    except Exception as e:
        print(f"Error: {e}")