import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob

def load_summary_files(base_path):
    """Load all summary.json files from the given directory structure."""
    model_data = {}
    model_folders = ['Llama70B', 'Llama8B', 'Llama1B']
    
    print(f"Looking for model folders in: {os.path.abspath(base_path)}")
    
    for model_folder in model_folders:
        folder_path = os.path.join(base_path, model_folder)
        print(f"\nChecking folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"  WARNING: Folder {folder_path} does not exist!")
            continue
            
        summary_files = glob.glob(os.path.join(folder_path, '**/summary.json'), recursive=True)
        
        if not summary_files:
            # Try direct path if no recursive match
            summary_file = os.path.join(folder_path, 'summary.json')
            if os.path.exists(summary_file):
                summary_files = [summary_file]
                print(f"  Found direct summary file: {summary_file}")
            else:
                print(f"  No summary.json files found in {folder_path}")
        else:
            print(f"  Found {len(summary_files)} summary files: {summary_files}")
        
        all_data = []
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    print(f"  Loaded data from {summary_file}")
                    print(f"    Keys in data: {list(data.keys())}")
                    all_data.append(data)
            except Exception as e:
                print(f"  ERROR loading {summary_file}: {e}")
        
        model_data[model_folder] = all_data
        print(f"  Total data entries for {model_folder}: {len(all_data)}")
    
    return model_data

def extract_method_data(data_dict, plot, method_prefix):
    """Extract all data for a specific method type (base, cove, dola, rag)."""
    method_data = []
    
    print(f"    Looking for method '{method_prefix}' in data keys: {list(data_dict.keys())}")
    
    for key, value in data_dict.items():
        print(f"      Checking key '{key}': type={value.get('type', 'N/A')}")
        if method_prefix in key and value.get('type') == method_prefix:
            if plot in value:
                method_data.append(value[plot])
                print(f"        Found data: {value[plot]}")
            else:
                print(f"        WARNING: '{plot}' not found in {key}")
    
    print(f"    Extracted {len(method_data)} entries for method '{method_prefix}'")
    return method_data

def calculate_mean_and_ci(data_list):
    """Calculate mean and 95% CI across multiple runs."""
    if not data_list:
        print("      No data to calculate mean and CI")
        return None, None, None
    
    print(f"      Calculating statistics for {len(data_list)} data points")
    print(f"      Data shapes: {[np.array(d).shape if isinstance(d, list) else 'scalar' for d in data_list]}")
    
    # Convert to numpy array for easier manipulation
    try:
        data_array = np.array(data_list)
        print(f"      Combined data array shape: {data_array.shape}")
        
        # Calculate mean across runs
        mean = np.mean(data_array, axis=0)
        
        # Calculate standard error
        std_err = stats.sem(data_array, axis=0)
        
        # Calculate 95% CI
        ci = 1.96 * std_err  # 95% CI for normal distribution
        
        print(f"      Mean: {mean}")
        print(f"      CI range: [{mean - ci}, {mean + ci}]")
        
        return mean, mean - ci, mean + ci
    except Exception as e:
        print(f"      ERROR in calculation: {e}")
        return None, None, None

def process_model_data(model_data, plot):
    """Process data for all models and methods."""
    processed_data = {}
    
    for model_name, data_list in model_data.items():
        print(f"\nProcessing model: {model_name}")
        processed_data[model_name] = {}
        
        if not data_list:
            print(f"  No data for {model_name}")
            continue
        
        # Combine all data from different files
        all_base = []
        all_cove = []
        all_dola = []
        all_rag = []
        
        for i, data_dict in enumerate(data_list):
            print(f"  Processing data file {i+1}/{len(data_list)}")
            
            # Extract data for each method
            base_data = extract_method_data(data_dict, plot, 'base')
            cove_data = extract_method_data(data_dict, plot, 'cove')
            dola_data = extract_method_data(data_dict, plot, 'dola')
            rag_data = extract_method_data(data_dict, plot, 'rag')
            
            all_base.extend(base_data)
            all_cove.extend(cove_data)
            all_dola.extend(dola_data)
            all_rag.extend(rag_data)
        
        print(f"  Total data points - Base: {len(all_base)}, CoVe: {len(all_cove)}, DoLa: {len(all_dola)}, RAG: {len(all_rag)}")
        
        # Calculate means and CIs
        print("  Calculating base statistics...")
        base_mean, base_lower, base_upper = calculate_mean_and_ci(all_base)
        print("  Calculating cove statistics...")
        cove_mean, cove_lower, cove_upper = calculate_mean_and_ci(all_cove)
        print("  Calculating dola statistics...")
        dola_mean, dola_lower, dola_upper = calculate_mean_and_ci(all_dola)
        print("  Calculating rag statistics...")
        rag_mean, rag_lower, rag_upper = calculate_mean_and_ci(all_rag)
        
        # Store processed data
        processed_data[model_name]['base'] = {
            'mean': base_mean,
            'lower': base_lower,
            'upper': base_upper
        }
        
        # Calculate relative differences from base (convert to percentage)
        if base_mean is not None and cove_mean is not None:
            try:
                cove_diff = (cove_mean - base_mean) * 100  # Convert to percentage (can be negative)
                # Propagate uncertainty for difference
                cove_diff_ci = np.sqrt((cove_upper - cove_mean)**2 + (base_upper - base_mean)**2) * 100
                processed_data[model_name]['cove'] = {
                    'diff': cove_diff,
                    'lower': cove_diff - cove_diff_ci,
                    'upper': cove_diff + cove_diff_ci
                }
                print(f"  CoVe difference calculated: {cove_diff}")
            except Exception as e:
                print(f"  ERROR calculating CoVe difference: {e}")
        
        if base_mean is not None and dola_mean is not None:
            try:
                dola_diff = (dola_mean - base_mean) * 100  # Convert to percentage (can be negative)
                dola_diff_ci = np.sqrt((dola_upper - dola_mean)**2 + (base_upper - base_mean)**2) * 100
                processed_data[model_name]['dola'] = {
                    'diff': dola_diff,
                    'lower': dola_diff - dola_diff_ci,
                    'upper': dola_diff + dola_diff_ci
                }
                print(f"  DoLa difference calculated: {dola_diff}")
            except Exception as e:
                print(f"  ERROR calculating DoLa difference: {e}")
        
        if base_mean is not None and rag_mean is not None:
            try:
                rag_diff = (rag_mean - base_mean) * 100  # Convert to percentage (can be negative)
                rag_diff_ci = np.sqrt((rag_upper - rag_mean)**2 + (base_upper - base_mean)**2) * 100
                processed_data[model_name]['rag'] = {
                    'diff': rag_diff,
                    'lower': rag_diff - rag_diff_ci,
                    'upper': rag_diff + rag_diff_ci
                }
                print(f"  RAG difference calculated: {rag_diff}")
            except Exception as e:
                print(f"  ERROR calculating RAG difference: {e}")
    
    return processed_data

def create_comparison_plot(processed_data, plot, method, output_file=None):
    """Create a comparison plot for a specific method."""
    print(f"\nCreating plot for method: {method}")
    
    plt.figure(figsize=(12, 8))  # Increased figure size to accommodate 4 models
    
    # Define colors and markers for each model (updated to include all 4 models)
    model_styles = {
        'Llama70B': {'color': 'blue', 'marker': 'o', 'label': 'Llama 70B'},
        'Llama8B': {'color': 'red', 'marker': 's', 'label': 'Llama 8B'},
        'Mistral7B': {'color': 'green', 'marker': '^', 'label': 'Mistral 7B'},
        'Llama1B': {'color': 'purple', 'marker': 'D', 'label': 'Llama 1B'}  # Added Llama1B
    }
    
    # Check if we have any data to plot
    has_data = False
    
    # Plot each model
    states = np.arange(6)  # 0 to 5
    
    for model_name, style in model_styles.items():
        if model_name in processed_data and method in processed_data[model_name]:
            data = processed_data[model_name][method]
            print(f"  Plotting data for {model_name}")
            print(f"    Diff shape: {np.array(data['diff']).shape}")
            print(f"    Diff values: {data['diff']}")
            
            # Ensure data is the right length
            if hasattr(data['diff'], '__len__') and len(data['diff']) == 6:
                # Plot line with error bars
                plt.errorbar(states, data['diff'], 
                            yerr=[data['diff'] - data['lower'], data['upper'] - data['diff']],
                            color=style['color'], 
                            marker=style['marker'],
                            markersize=8,
                            linewidth=2,
                            capsize=3,
                            capthick=1,
                            elinewidth=1,
                            alpha=0.7,
                            label=style['label'])
                has_data = True
            else:
                print(f"    WARNING: Data length mismatch for {model_name}. Expected 6, got {len(data['diff']) if hasattr(data['diff'], '__len__') else 'scalar'}")
        else:
            print(f"  No data found for {model_name} - {method}")
    
    if not has_data:
        print(f"  WARNING: No data to plot for method {method}")
        plt.text(0.5, 0.5, f'No data available for {method}', 
                transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)
    
    # Customize plot
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Relative Difference from Base (%)', fontsize=14)
    plt.title(f'{method.upper()} vs Base Comparison Across Models for {plot.replace("_", " ").title()}', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 5.5)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set x-axis ticks
    plt.xticks(states)
    
    plt.tight_layout()
    
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {output_file}")
    else:
        plt.show()

# Main execution
if __name__ == "__main__":
    # Set the base path where your model folders are located
    base_path = "datasets/CodeForce/neoresults"  # Adjust this to your actual path
    plot = 'instruction_hallucination'
    
    # Load all summary files
    print("Loading summary files...")
    model_data = load_summary_files(base_path)
    
    if not any(model_data.values()):
        print("ERROR: No data loaded! Check your file structure and paths.")
        exit(1)
    
    # Process the data
    print("\nProcessing data...")
    processed_data = process_model_data(model_data, plot)
    
    # Print summary of processed data
    print("\nProcessed data summary:")
    for model_name, model_info in processed_data.items():
        print(f"  {model_name}: {list(model_info.keys())}")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Create plots
    print("\nCreating plots...")
    
    # CoVe plot
    create_comparison_plot(processed_data, plot, 'cove', f'plots/plots_llamas/{plot}/cove_comparison.png')
    
    # DoLa plot
    create_comparison_plot(processed_data, plot, 'dola', f'plots/plots_llamas/{plot}/dola_comparison.png')
    
    # RAG plot
    create_comparison_plot(processed_data, plot, 'rag', f'plots/plots_llamas/{plot}/rag_comparison.png')
    
    print("\nScript completed!")