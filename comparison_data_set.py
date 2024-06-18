import csv
import json
import re

# Function to read and parse the CSV file
def read_csv(file_path):
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # Skip the header
        for row in reader:
            # Join the row elements to handle potential delimiters within the sequence string
            row_str = ''.join(row)
            # Find the part of the row that contains the sequence list
            sequences_str = re.findall(r'\[\[.*\]\]', row_str)
            if sequences_str:
                sequences = eval(sequences_str[0])  # Evaluate the string to convert it to a list
                # Extract only the start and end positions
                cleaned_sequences = [[seq[0], seq[1]] for seq in sequences]
                data.append(cleaned_sequences)
            else:
                print(f"No sequences found in row: {row_str}")
    return data

# Function to read and parse the JSONL file
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as jsonlfile:
        for line in jsonlfile:
            json_obj = json.loads(line)
            sequences = json_obj.get("label", [])
            # Extract only the start and end positions
            cleaned_sequences = [[seq[0], seq[1]] for seq in sequences]
            data.append(cleaned_sequences)
    return data

# Function to compare the sequences
def compare_sequences(csv_data, jsonl_data):
    differences = []
    count_dif = 0
    for idx, (csv_seq, jsonl_seq) in enumerate(zip(csv_data, jsonl_data)):
        if csv_seq != jsonl_seq:
            differences.append((idx, csv_seq, jsonl_seq))
            count_dif+=1

    return differences, count_dif

# Main function
def main(csv_file_path, jsonl_file_path):
    csv_data = read_csv(csv_file_path)
    jsonl_data = read_jsonl(jsonl_file_path)
    
    differences, count_dif = compare_sequences(csv_data, jsonl_data)
    
    if differences:
        print("Differences found:")
        for idx, csv_seq, jsonl_seq in differences:
            print(f"Row {idx + 1}:")
            print(f"CSV: {csv_seq}")
            print(f"JSONL: {jsonl_seq}")
        
        print("The following number of sentences do not match:")
        print(count_dif)
    else:
        print("No differences found. The files match.")

# Example usage
csv_file_path = 'Social_Group_mentions.csv'
jsonl_file_path = 'mathieu.jsonl'

main(csv_file_path, jsonl_file_path)
