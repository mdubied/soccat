import json
import matplotlib.pyplot as plt
from collections import Counter

# Function to count group occurrences in the JSONL file
def count_group_occurrences(file_path):
    group_count = Counter()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            groups = json_obj.get('groups', [])
            for group in groups:
                group_count[group] += 1
    return group_count

# Main function
def main(input_file, output_file, title, figure_size_cm, max_percentage, category_font_size, adjust_top,  highlight_categories, max_categories=None, show=True):
    group_count = count_group_occurrences(input_file)
    total_count = sum(group_count.values())

    # Calculate relative frequencies in percent
    relative_frequencies = {group: (count / total_count) * 100 for group, count in group_count.items()}

    # Sort the group count for better visualization
    sorted_groups = sorted(relative_frequencies.items(), key=lambda item: item[1], reverse=True)
    
    # Limit the number of categories to display
    if max_categories is not None:
        if max_categories and max_categories > 0:
            sorted_groups = sorted_groups[:max_categories]

    # Prepare data for the bar chart
    group_names = [item[0] for item in sorted_groups]
    percentages = [item[1] for item in sorted_groups]

    # Convert figure size from cm to inches
    figure_size_inches = (figure_size_cm[0] / 2.54, figure_size_cm[1] / 2.54)

    # Create a horizontal bar chart
    plt.figure(figsize=figure_size_inches)
    bars = plt.barh(group_names, percentages, color='skyblue')

    # Highlight specific categories
    for bar, group_name in zip(bars, group_names):
        if group_name in highlight_categories:
            bar.set_color('orange')
            plt.gca().get_yticklabels()[group_names.index(group_name)].set_fontstyle('italic')

    plt.yticks(fontsize=category_font_size)
    plt.xlabel('Relative frequency (%)', fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlim(0, max_percentage)
    plt.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.7, zorder=0)
    plt.gca().invert_yaxis()  # To have the longest bars on top
    plt.tight_layout()
    plt.title(title, fontsize=12)
    plt.subplots_adjust(top=adjust_top)

    # Save figure
    plt.savefig(output_file, format='pdf')

    if show:
        plt.show()

# Example usage
# Choose which combination
language = 'FG' 
level = 'lowest'

# File path
input_file = 'group_per_sentence/groups_per_sentence_' + language + '_' + level + '_level.jsonl'
output_file = 'figures/results_cat_histo_'+ language + '_' + level + '_level.pdf'

# Title of plot
if language == 'F':
    title_plot = 'French newspapers'
elif language == 'G':
    title_plot = 'German newspapers'
else:
    title_plot = 'French and German newspapers'

# Size of plot elements
if level == 'highest':
    figure_size_cm = (17,10)
    adjust_top = 0.9
    max_percentage = 35
    category_font_size = 10
else:
    figure_size_cm = (17,27)
    adjust_top = 0.96
    max_percentage = 35
    category_font_size = 8

# Categories to highlight
cat_in_colors = [
    'politicians and high-ranking officials',
    'soldiers',
    'Other',
    'journalistes',
    'minors',
    'teachers and educators',
    'employees',
    'unemployed',
    'patients',
    'health and care professionals',
    'scientists',
    'women',
    'men',
    'immigrants',
    'authors and artists',
    'tax payers',
    'large enterprises',
    'investors and stakeholders',
    'athletes'
]

# Number of categories to display
max_categories = None
# max_categories = 25
# figure_size_cm = (17,20)

# Run the main function
main(input_file, output_file, title_plot, figure_size_cm, max_percentage, category_font_size, adjust_top, 
     cat_in_colors, max_categories=max_categories, show=False)
