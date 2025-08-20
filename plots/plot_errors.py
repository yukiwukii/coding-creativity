import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re

def truncate_error_message(error_message):
    """
    Truncate error messages according to specified rules.
    For TypeError with missing positional arguments, truncate after 'positional argument'
    """
    # Pattern for TypeError missing positional argument
    type_error_pattern = r'(TypeError: .+ missing \d+ required positional argument)'
    
    match = re.search(type_error_pattern, error_message)
    if match:
        return match.group(1)
    
    # Add more truncation rules here if needed
    # For now, return the original message for other error types
    return error_message

def load_and_process_data(json_file_path):
    """Load JSON data and process error messages"""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract and process error messages
    errors = data['errors']
    processed_errors = []
    
    for error in errors:
        processed_error = error.copy()
        processed_error['truncated_error_message'] = truncate_error_message(error['error_message'])
        processed_errors.append(processed_error)
    
    return processed_errors

def plot_error_types(model_name, errors, plot_type='bar'):
    """Create plots for error type analysis"""
    
    # Count error types
    error_type_counts = Counter([error['error_type'] for error in errors])
    
    # Count truncated error messages
    error_message_counts = Counter([error['truncated_error_message'] for error in errors])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Error Analysis Dashboard for {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Error Types Bar Chart
    error_types = list(error_type_counts.keys())
    error_type_values = list(error_type_counts.values())
    
    bars1 = ax1.bar(error_types, error_type_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Distribution of Error Types', fontweight='bold')
    ax1.set_xlabel('Error Type')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. Top Error Messages
    top_errors = error_message_counts.most_common(10)
    error_messages = [msg[:50] + '...' if len(msg) > 50 else msg for msg, _ in top_errors]
    error_counts = [count for _, count in top_errors]
    
    bars2 = ax2.barh(range(len(error_messages)), error_counts, color='#FF9F43')
    ax2.set_title('Top 10 Error Messages (Truncated)', fontweight='bold')
    ax2.set_xlabel('Count')
    ax2.set_yticks(range(len(error_messages)))
    ax2.set_yticklabels(error_messages, fontsize=8)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center')
    
    # 3. Top 5 Error Messages Pie Chart (instead of error types)
    top_5_errors = error_message_counts.most_common(5)
    other_count = sum(error_message_counts.values()) - sum(count for _, count in top_5_errors)
    
    # Prepare data for pie chart
    pie_labels = []
    pie_values = []
    pie_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F', '#BB8FCE']
    
    for i, (msg, count) in enumerate(top_5_errors):
        # Truncate label for readability
        label = msg[:30] + '...' if len(msg) > 30 else msg
        pie_labels.append(label)
        pie_values.append(count)
    
    # Add "Others" category if there are more errors
    if other_count > 0:
        pie_labels.append('Others')
        pie_values.append(other_count)
    
    # Create pie chart with better formatting
    wedges, texts, autotexts = ax3.pie(pie_values, labels=None, autopct='%1.1f%%', 
                                      colors=pie_colors[:len(pie_values)], 
                                      startangle=90, textprops={'fontsize': 8})
    
    # Create legend outside the pie chart
    total_errors = sum(pie_values)
    legend_labels = []
    for i, (label, value) in enumerate(zip(pie_labels, pie_values)):
        percentage = (value / total_errors) * 100
        legend_labels.append(f'{label}\n({value}, {percentage:.1f}%)')
    
    ax3.legend(wedges, legend_labels, title="Error Messages", loc="center left", 
               bbox_to_anchor=(1, 0, 0.5, 1), fontsize=7)
    ax3.set_title('Top 5 Error Messages Distribution', fontweight='bold')
    
    # 4. Problem ID distribution (top 10)
    problem_ids = [error['problem_id'] for error in errors]
    problem_id_counts = Counter(problem_ids)
    top_problems = problem_id_counts.most_common(10)
    
    if top_problems:
        problem_names = [prob for prob, _ in top_problems]
        problem_counts = [count for _, count in top_problems]
        
        bars4 = ax4.bar(problem_names, problem_counts, color='#E74C3C')
        ax4.set_title('Top 10 Problem IDs with Most Errors', fontweight='bold')
        ax4.set_xlabel('Problem ID')
        ax4.set_ylabel('Error Count')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def print_error_summary(errors):
    """Print summary statistics"""
    print("=" * 50)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 50)
    
    total_errors = len(errors)
    print(f"Total errors: {total_errors}")
    
    # Error type breakdown
    error_type_counts = Counter([error['error_type'] for error in errors])
    print(f"\nError type breakdown:")
    for error_type, count in error_type_counts.most_common():
        percentage = (count / total_errors) * 100
        print(f"  {error_type}: {count} ({percentage:.1f}%)")
    
    # Most common error messages
    error_message_counts = Counter([error['truncated_error_message'] for error in errors])
    print(f"\nTop 5 most common error messages:")
    for i, (msg, count) in enumerate(error_message_counts.most_common(5), 1):
        percentage = (count / total_errors) * 100
        print(f"  {i}. {msg[:80]}{'...' if len(msg) > 80 else ''}")
        print(f"     Count: {count} ({percentage:.1f}%)")
    
    # Test mode breakdown
    test_modes = [error['test_mode'] for error in errors]
    test_mode_counts = Counter(test_modes)
    print(f"\nTest mode breakdown:")
    for mode, count in test_mode_counts.most_common():
        percentage = (count / total_errors) * 100
        print(f"  {mode}: {count} ({percentage:.1f}%)")

# Main execution
if __name__ == "__main__":
    # Replace 'your_file.json' with the path to your JSON file
    model_name = 'Llama1B'
    json_file_path = f'datasets/CodeForce/evaluation/{model_name}/errors/merged_ALL.json'
    
    try:
        # Load and process data
        print("Loading and processing data...")
        errors = load_and_process_data(json_file_path)
        
        # Print summary
        print_error_summary(errors)
        
        # Create plots
        print("\nCreating visualizations...")
        fig = plot_error_types(model_name, errors)
        
        # Save the plot
        plt.savefig(f'plots/plot_errors/{model_name}.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'error_analysis.png'")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file_path}'")
        print("Please make sure the JSON file exists in the current directory.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")