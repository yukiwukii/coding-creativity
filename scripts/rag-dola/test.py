import json

# Load the data files
with open('Mistral7B/Base1.json','r') as file:
    data = json.load(file)
with open('Mistral7B/Base2.json','r') as file:
    data2 = json.load(file)
with open('Mistral7B/Base3.json', 'r') as file:
    data3 = json.load(file)

print("=" * 80)
print("INSPECTING OUTPUTS FROM JSON FILES")
print("=" * 80)

# Function to safely print outputs
def print_outputs(data_list, file_name, max_items=3):
    print(f"\n--- {file_name} ---")
    
    for idx, item in enumerate(data_list[:max_items]):  # Only check first few items
        print(f"\nIndex {idx}:")
        
        # Check if outputs exist
        if "outputs" not in item:
            print("  ❌ No 'outputs' key found")
            continue
            
        outputs = item["outputs"]
        print(f"  📊 Outputs type: {type(outputs)}")
        
        if outputs is None:
            print("  ❌ Outputs is None")
        elif isinstance(outputs, list):
            print(f"  📝 Number of outputs: {len(outputs)}")
            
            if len(outputs) == 0:
                print("  ⚠️  Empty outputs list")
            else:
                for out_idx, output in enumerate(outputs):
                    print(f"\n  Output {out_idx}:")
                    print(f"    Type: {type(output)}")
                    if isinstance(output, str):
                        print(f"    Length: {len(output)} characters")
                        if output.strip():
                            # Print first 200 characters
                            preview = output.strip()[:200]
                            print(f"    Preview: {repr(preview)}")
                            if len(output.strip()) > 200:
                                print("    ... (truncated)")
                        else:
                            print("    ⚠️  Empty string")
                    else:
                        print(f"    Value: {output}")
        else:
            print(f"  ⚠️  Outputs is not a list: {outputs}")

# Print outputs from all three files
print_outputs(data, "Base1.json")
print_outputs(data2, "Base2.json") 
print_outputs(data3, "Base3.json")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

files_data = [("Base1.json", data), ("Base2.json", data2), ("Base3.json", data3)]

for file_name, file_data in files_data:
    total_items = len(file_data)
    items_with_outputs = 0
    items_with_valid_outputs = 0
    
    for item in file_data:
        if "outputs" in item and item["outputs"]:
            items_with_outputs += 1
            if (isinstance(item["outputs"], list) and 
                len(item["outputs"]) > 0 and 
                any(output and str(output).strip() for output in item["outputs"])):
                items_with_valid_outputs += 1
    
    print(f"\n{file_name}:")
    print(f"  Total items: {total_items}")
    print(f"  Items with outputs key: {items_with_outputs}")
    print(f"  Items with valid outputs: {items_with_valid_outputs}")
    print(f"  Items still need processing: {total_items - items_with_valid_outputs}")

# Check specific indices if you want
print("\n" + "=" * 80)
print("CHECK SPECIFIC INDICES")
print("=" * 80)

# Modify these indices to check specific items
indices_to_check = [0, 1, 2]  # Change this to the indices you're interested in

for idx in indices_to_check:
    if idx < len(data):
        print(f"\nDetailed check for index {idx}:")
        
        for file_name, file_data in files_data:
            item = file_data[idx]
            has_outputs = "outputs" in item
            outputs_value = item.get("outputs")
            
            print(f"  {file_name}:")
            print(f"    Has outputs key: {has_outputs}")
            print(f"    Outputs value: {outputs_value}")
            
            if has_outputs and isinstance(outputs_value, list) and len(outputs_value) > 0:
                print(f"    First output (first 100 chars): {repr(str(outputs_value[0])[:100])}")
    else:
        print(f"Index {idx} is out of range (max index: {len(data)-1})")