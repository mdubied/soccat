import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Function to count group occurrences in the Excel file
def count_group_occurrences(file_path):
    group_count = Counter()
    df = pd.read_excel(file_path)

    # Handle NaN values in 'main_group' column
    df['main_group'] = df['main_group'].fillna('')  # Replace NaN with empty string

    for _, row in df.iterrows():
        groups = row['main_group'].replace("#N/A", "").split('&')
        count = row['Absolute_Number']
        for group in groups:
            group_count[group.strip()] += count

    return group_count

# Main function
def main(input_files, output_file, title, figure_size_cm, max_percentage, category_font_size, adjust_top, highlight_categories, max_categories=None, show=True):
    group_count = Counter()
    
    # Count group occurrences from both input files
    for file in input_files:
        group_count.update(count_group_occurrences(file))

    # Calculate total count excluding empty string category
    total_count = sum(count for group, count in group_count.items() if group != '')

    # Calculate relative frequencies excluding empty string category
    relative_frequencies = {group: (count / total_count) * 100 for group, count in group_count.items() if group != ''}

    # Sort the group count for better visualization
    sorted_groups = sorted(relative_frequencies.items(), key=lambda item: item[1], reverse=True)
    
    # Limit the number of categories to display
    if max_categories is not None and max_categories > 0:
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
input_files = ['proportion_group_positive.xlsx', 'proportion_group_negative.xlsx']
output_file = 'figures/frequency_manifestos.pdf'

# Title of plot
title_plot = 'Frequency of Group Mentions in German Manifestos'

# Size of plot elements
figure_size_cm = (17, 20)
adjust_top = 0.96
max_percentage = 20
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
max_categories = 25

# Run the main function
main(input_files, output_file, title_plot, figure_size_cm, max_percentage, category_font_size, adjust_top, 
     cat_in_colors, max_categories=max_categories, show=False)
