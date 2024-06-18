import json
import matplotlib.pyplot as plt
from collections import Counter

# Function to read and extract categories from label_config.json
def read_categories(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    categories = [item['text'] for item in json_data if item['text'] != "Other"]
    return categories

# Function to count category occurrences in mathieu.jsonl
def count_category_occurrences(file_path, categories):
    category_count = Counter()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            labels = json_obj.get('label', [])
            for label in labels:
                category_name = label[2]  # The third element in each label is the category name
                if category_name in categories:
                    category_count[category_name] += 1
    return category_count

# Main function
def main(label_config_path, mathieu_path, figure_size_cm, output_file):
    categories = read_categories(label_config_path)
    category_count = count_category_occurrences(mathieu_path, categories)

    # Sort the category count for better visualization
    sorted_categories = sorted(category_count.items(), key=lambda item: item[1], reverse=True)
    
    # Prepare data for the bar chart
    category_names = [item[0] for item in sorted_categories]
    counts = [item[1] for item in sorted_categories]

    # Convert figure size from cm to inches
    figure_size_inches = (figure_size_cm[0] / 2.54, figure_size_cm[1] / 2.54)

    # Create a horizontal bar chart
    plt.figure(figsize=figure_size_inches)
    plt.barh(category_names, counts, color='skyblue')
    plt.yticks(fontsize=8)
    plt.xlabel('Number of Occurrences')
    plt.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.gca().invert_yaxis()  # To have the longest bars on top
    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.show()




# File paths
label_config_path = 'label_config.json'
doccano_export = 'specific_classification_doccano_german.jsonl'

# Figure size in cm (width, height)
figure_size_cm = (19, 27)

# Output file name
output_file = 'results_cat_histo_german.pdf'

# Run the main function
main(label_config_path, doccano_export, figure_size_cm, output_file)
