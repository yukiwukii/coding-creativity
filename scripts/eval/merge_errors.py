import json
import os
import glob
from collections import defaultdict
import re

def should_prioritize_error(error_message):
    """
    Returns True if this error should be prioritized over ValueError: invalid literal for int() with base 10
    """
    return "ValueError: invalid literal for int() with base 10:" not in error_message

def extract_source_from_filename(filename):
    """
    Extract the source identifier from filename like:
    Llama-1B-Instruct_sample=199_dp=5_base1_errors.json -> base1
    """
    # Get just the filename without path
    basename = os.path.basename(filename)
    # Remove the .json extension
    name_without_ext = basename.replace('.json', '')
    
    # Look for pattern: _[source]_errors at the end
    match = re.search(r'_([^_]+)_errors$', name_without_ext)
    if match:
        return match.group(1)
    
    # Fallback: if no match, return the filename without extension
    return name_without_ext

def merge_json_files(folder_path):
    """
    Merge all JSON files in the specified folder according to the merging rules.
    """
    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print("No JSON files found in the specified folder.")
        return
    
    # Dictionary to store merged errors: (problem_id, code_snippet) -> error_entry
    merged_errors = {}
    
    # Process each JSON file
    for file_path in json_files:
        # Skip if it's already a merged file
        if file_path.endswith("_ALL.json"):
            continue
            
        print(f"Processing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract errors from the JSON
            errors = data.get('errors', [])
            
            # Extract source from filename
            source = extract_source_from_filename(file_path)
            
            for error in errors:
                problem_id = error.get('problem_id')
                code_snippet = error.get('code_snippet')
                error_message = error.get('error_message', '')
                
                # Add source to the error entry
                error_with_source = error.copy()
                error_with_source['source'] = source
                
                # Create a unique key for this problem_id + code_snippet combination
                key = (problem_id, code_snippet)
                
                if key not in merged_errors:
                    # First occurrence of this combination
                    merged_errors[key] = error_with_source
                else:
                    # We have a duplicate - apply merging logic
                    existing_error = merged_errors[key]
                    existing_message = existing_error.get('error_message', '')
                    
                    # Priority logic:
                    # 1. Prioritize any error that's NOT the ValueError
                    # 2. If both are ValueError or both are not ValueError, keep the existing one
                    
                    current_is_value_error = "ValueError: invalid literal for int() with base 10:" in error_message
                    existing_is_value_error = "ValueError: invalid literal for int() with base 10:" in existing_message
                    
                    # If current error is not a ValueError and existing is ValueError, replace
                    if not current_is_value_error and existing_is_value_error:
                        merged_errors[key] = error_with_source
                    # If current is ValueError and existing is not ValueError, keep existing
                    elif current_is_value_error and not existing_is_value_error:
                        pass  # Keep existing
                    # If both are the same type (both ValueError or both not ValueError), keep existing
                    else:
                        pass  # Keep existing
                        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {file_path}: {e}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Create the merged JSON structure
    merged_data = {
        "errors": list(merged_errors.values())
    }
    
    # Sort errors by error_id if available, otherwise by problem_id
    merged_data["errors"].sort(key=lambda x: (x.get('error_id', 0), x.get('problem_id', '')))
    
    # Write the merged file
    output_path = os.path.join(folder_path, "merged_ALL.json")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMerged {len(json_files)} JSON files into {output_path}")
        print(f"Total unique error entries: {len(merged_data['errors'])}")
        
    except Exception as e:
        print(f"Error writing merged file: {e}")

def main():
    """
    Main function to run the merger.
    """
    # You can specify the folder path here, or modify to accept command line arguments
    folder_path = input("Enter the folder path containing JSON files: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Folder path '{folder_path}' does not exist.")
        return
    
    if not os.path.isdir(folder_path):
        print(f"'{folder_path}' is not a directory.")
        return
    
    merge_json_files(folder_path)

if __name__ == "__main__":
    main()