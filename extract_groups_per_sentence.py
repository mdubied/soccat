import json

def transform_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for row_idx, line in enumerate(infile):
            data = json.loads(line)
            transformed_data = {
                "id": data["id"],
                "rowIdx": row_idx,
                "groups": [group[2] for group in data["label"]],
                "text": data["text"],
            }
            outfile.write(json.dumps(transformed_data, ensure_ascii=False) + '\n')

# Example usage:
input_file = 'specific_classification_doccano_german.jsonl'
output_file = 'groups_per_sentence_G_lowest_level.jsonl'
transform_jsonl(input_file, output_file)
