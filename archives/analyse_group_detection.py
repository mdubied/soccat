import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_group_lengths_histogram(group_lengths, output_file, figsize, show=True):
    # Convert figure size from cm to inches
    figure_size_inches = (figsize[0] / 2.54, figsize[1] / 2.54)

    # Plot
    plt.figure(figsize=figure_size_inches)
    plt.hist(group_lengths, bins=range(1, max(group_lengths) + 1), edgecolor='black', zorder=5)
    plt.xlabel('Group Mention Length (characters)',fontsize=10)
    plt.ylabel('Frequency',fontsize=10)
    plt.xticks(fontsize=10)
    # plt.title('Distribution of Group Mention Lengths',fontsize=10)
    plt.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig(output_file)

    # Save figure
    plt.savefig(output_file, format='pdf')

    if show == True:
        plt.show()


def process_file(file_path):
    row_counts = defaultdict(int)
    total_group_length = 0
    total_group_count = 0
    group_lengths = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            labels = json_obj.get('label', [])
            group_count = len(labels)
            
            # Update row counts
            if group_count <= 10:
                row_counts[group_count] += 1
            else:
                row_counts[10] += 1
            
            # Calculate total group length and store each group length
            for label in labels:
                start, end, _ = label
                length = end - start + 1
                total_group_length += length
                total_group_count += 1
                group_lengths.append(length)

    # Compute mean group length
    mean_group_length = total_group_length / total_group_count if total_group_count > 0 else 0

    # Calculate total lines
    total_lines = sum(row_counts.values())

    return row_counts, mean_group_length, total_lines, group_lengths

def process_files(file_list):
    results = {}
    total_group_counts = defaultdict(int)
    total_group_length = 0
    total_group_count = 0
    total_lines = 0
    all_group_lengths = []
    
    for file_path in file_list:
        row_counts, mean_group_length, num_lines, group_lengths = process_file(file_path)

        # Accumulate totals for the group
        for count, num in row_counts.items():
            total_group_counts[count] += num
        total_group_length += mean_group_length * num_lines
        total_group_count += sum(row_counts.values())
        total_lines += num_lines
        all_group_lengths.extend(group_lengths)
        
        results[file_path] = {
            'row_counts': row_counts,
            'mean_group_length': mean_group_length,
            'total_lines': num_lines,
            'group_lengths': group_lengths
        }

    # Compute overall mean group length for the group
    overall_mean_group_length = total_group_length / total_lines if total_lines > 0 else 0

    # Store group totals
    results['group_totals'] = {
        'row_counts': total_group_counts,
        'mean_group_length': overall_mean_group_length,
        'total_lines': total_lines,
        'group_lengths': all_group_lengths
    }

    return results


def compute_percentages(row_counts, total_lines):
    percentages = {}
    for i in range(3):
        percentages[f'{i} group mentions'] = (row_counts[i] / total_lines * 100) if total_lines > 0 else 0
    # For 3 or more group mentions
    more_than_3 = sum(row_counts[i] for i in range(3, 11))
    percentages['3 or more group mentions'] = (more_than_3 / total_lines * 100) if total_lines > 0 else 0
    return percentages

def generate_latex_table(results, group1_results, group2_results, output_file):
    # Define table headers
    table_header = r"""
    \begin{table}[htbp]
    \centering
    \begin{tabular}{lcccc}
    \toprule
    Newspapers & 0 group mention (\%) & 1 group mention (\%) & 2 groups mention (\%) & 3 or more groups (\%) \\
    \midrule
    """
    
    # Initialize table content
    table_content = ""

    # Process individual files
    for file_path, result in results.items():
        if file_path != 'group_totals':  # Exclude group_totals entry
            row_counts = result['row_counts']
            total_lines = result['total_lines']
            percentages = compute_percentages(row_counts, total_lines)
            table_content += f"{file_path} & {percentages['0 group mentions']:.0f} & {percentages['1 group mentions']:.0f} & {percentages['2 group mentions']:.0f} & {percentages['3 or more group mentions']:.0f} \\\\\n"

    # Process group1 (French)
    group1_totals = group1_results['group_totals']
    group1_percentages = compute_percentages(group1_totals['row_counts'], group1_totals['total_lines'])
    table_content += f"Group 1 (French) & {group1_percentages['0 group mentions']:.0f} & {group1_percentages['1 group mentions']:.0f} & {group1_percentages['2 group mentions']:.0f} & {group1_percentages['3 or more group mentions']:.0f} \\\\\n"

    # Process group2 (German)
    group2_totals = group2_results['group_totals']
    group2_percentages = compute_percentages(group2_totals['row_counts'], group2_totals['total_lines'])
    table_content += f"Group 2 (German) & {group2_percentages['0 group mentions']:.0f} & {group2_percentages['1 group mentions']:.0f} & {group2_percentages['2 group mentions']:.0f} & {group2_percentages['3 or more group mentions']:.0f} \\\\\n"

    # Process overall group
    overall_totals = results['group_totals']
    overall_percentages = compute_percentages(overall_totals['row_counts'], overall_totals['total_lines'])
    table_content += f"All files & {overall_percentages['0 group mentions']:.0f} & {overall_percentages['1 group mentions']:.0f} & {overall_percentages['2 group mentions']:.0f} & {overall_percentages['3 or more group mentions']:.0f} \\\\\n"

    # Table footer
    table_footer = r"""
    \bottomrule
    \end{tabular}
    \caption{Group Mentions in Newspapers}
    \label{tab:group_mentions}
    \end{table}
    """

    # Combine all parts into the final LaTeX table
    latex_table = table_header + table_content + table_footer

    # Write the LaTeX table to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)


# List of JSONL files
file_list = glob.glob('has_group_files/french/*.jsonl') + glob.glob('has_group_files/german/*.jsonl')  # Adjust the path as needed

# Define two groups of files
group1_files = glob.glob('has_group_files/french/*.jsonl')
group2_files = glob.glob('has_group_files/german/*.jsonl')

# Process files and get results
results = process_files(file_list)

# Process each subgroup separately
group1_results = process_files(group1_files)
group2_results = process_files(group2_files)

# Display results for individual files
for file_path, result in results.items():
    if file_path != 'group_totals':
        print(f"Results for {file_path}:")
        print("Row counts with specific group mentions:")
        # for i in range(11):
        #     print(f"{i} group mentions: {result['row_counts'][i]} rows")
        print(f"Total number of lines: {result['total_lines']}")
        print(f"Mean group length: {result['mean_group_length']:.2f}")
        percentages = compute_percentages(result['row_counts'], result['total_lines'])
        for key, value in percentages.items():
            print(f"{key}: {value:.2f}%")
        print()

# Display results for the overall group
print("Results for the overall group:")
overall_totals = results['group_totals']
# for i in range(11):
#     print(f"{i} group mentions: {overall_totals['row_counts'][i]} rows")
print(f"Total number of lines: {overall_totals['total_lines']}")
print(f"Mean group length: {overall_totals['mean_group_length']:.2f}")
percentages = compute_percentages(overall_totals['row_counts'], overall_totals['total_lines'])
for key, value in percentages.items():
    print(f"{key}: {value:.2f}%")
print()

# Display results for group 1
print("Results for group 1 (French):")
group1_totals = group1_results['group_totals']
# for i in range(11):
#     print(f"{i} group mentions: {group1_totals['row_counts'][i]} rows")
print(f"Total number of lines: {group1_totals['total_lines']}")
print(f"Mean group length: {group1_totals['mean_group_length']:.2f}")
percentages = compute_percentages(group1_totals['row_counts'], group1_totals['total_lines'])
for key, value in percentages.items():
    print(f"{key}: {value:.2f}%")
print()

# Display results for group 2
print("Results for group 2 (German):")
group2_totals = group2_results['group_totals']
# for i in range(11):
#     print(f"{i} group mentions: {group2_totals['row_counts'][i]} rows")
print(f"Total number of lines: {group2_totals['total_lines']}")
print(f"Mean group length: {group2_totals['mean_group_length']:.2f}")
percentages = compute_percentages(group2_totals['row_counts'], group2_totals['total_lines'])
for key, value in percentages.items():
    print(f"{key}: {value:.2f}%")
print()

# Call the function to generate the LaTeX table and store it in a text file
generate_latex_table(results, group1_results, group2_results, 'group_mentions_table.tex')

# Collect all group lengths
all_group_lengths = results['group_totals']['group_lengths']

# Plot histogram
plot_group_lengths_histogram(all_group_lengths, 'group_lengths_histogram.pdf',(17,6))
