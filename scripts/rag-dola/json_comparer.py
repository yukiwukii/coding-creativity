import json

def compare_json_outputs(file1_path, file2_path):
    """
    Compare outputs field in two JSON files and find where they start to be the same.
    
    Args:
        file1_path (str): Path to first JSON file
        file2_path (str): Path to second JSON file
    """
    
    # Load both JSON files
    try:
        with open(file1_path, 'r') as f1:
            data1 = json.load(f1)
        with open(file2_path, 'r') as f2:
            data2 = json.load(f2)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return
    
    # Handle different JSON structures
    # If the JSON is a list of objects
    if isinstance(data1, list) and isinstance(data2, list):
        items1 = data1
        items2 = data2
    # If the JSON has a top-level object with arrays
    elif isinstance(data1, dict) and isinstance(data2, dict):
        # Try to find the array in the JSON structure
        # Adjust these keys based on your JSON structure
        if 'items' in data1:
            items1 = data1['items']
            items2 = data2['items']
        else:
            # If it's a single object, wrap in list
            items1 = [data1]
            items2 = [data2]
    else:
        print("Error: Unexpected JSON structure")
        return
    
    # Compare lengths
    min_length = min(len(items1), len(items2))
    max_length = max(len(items1), len(items2))
    
    print(f"File 1 has {len(items1)} items")
    print(f"File 2 has {len(items2)} items")
    print(f"Comparing first {min_length} items")
    print("-" * 50)
    
    # Find where outputs start to be the same
    same_from_index = None
    first_difference = None
    differences = []
    
    for idx in range(min_length):
        try:
            outputs1 = items1[idx].get("outputs", [])
            outputs2 = items2[idx].get("outputs", [])
            
            # Get problem_id for reference (if available)
            problem_id1 = items1[idx].get("problem_id", f"item_{idx}")
            problem_id2 = items2[idx].get("problem_id", f"item_{idx}")
            
            if outputs1 == outputs2:
                if same_from_index is None:
                    same_from_index = idx
                    print(f"✓ Outputs start being the same from index {idx} (problem_id: {problem_id1})")
                    break
            else:
                if first_difference is None:
                    first_difference = idx
                
                differences.append({
                    'index': idx,
                    'problem_id1': problem_id1,
                    'problem_id2': problem_id2,
                    'outputs1': outputs1,
                    'outputs2': outputs2
                })
                
                print(f"✗ Index {idx} - Different outputs")
                print(f"    Problem ID 1: {problem_id1}")
                print(f"    Problem ID 2: {problem_id2}")
                print(f"    Outputs 1: {outputs1}")
                print(f"    Outputs 2: {outputs2}")
                print()
        
        except (KeyError, AttributeError) as e:
            print(f"Error at index {idx}: {e}")
            continue
    
    # Summary
    print("-" * 50)
    print("SUMMARY:")
    
    if same_from_index is not None:
        print(f"✓ Outputs are the same starting from index: {same_from_index}")
        if same_from_index == 0:
            print("  All compared outputs are identical!")
    else:
        print("✗ No identical outputs found in the compared range")
    
    if first_difference is not None:
        print(f"✗ First difference found at index: {first_difference}")
    
    print(f"Total differences found: {len(differences)}")
    
    # Check remaining items if files have different lengths
    if len(items1) != len(items2):
        print(f"\nNote: Files have different lengths ({len(items1)} vs {len(items2)})")
        longer_file = "File 1" if len(items1) > len(items2) else "File 2"
        print(f"{longer_file} has {abs(len(items1) - len(items2))} additional items")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    file1 = "raginftemplate.json"  # Replace with your first JSON file path
    file2 = "datasets/CodeForce/inference/Llama70B/Rag1.json"  # Replace with your second JSON file path
    
    compare_json_outputs(file1, file2)