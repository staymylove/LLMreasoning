import json
import os
from tqdm import tqdm
import random

def merge_jsonl_files(input_files, output_file):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Store all records
    all_records = []
    file_counts = {}
    
    # Read each input file
    for input_file in input_files:
        try:
            file_records = []
            # Count lines first for the progress bar
            with open(input_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=num_lines, desc=f"Reading {os.path.basename(input_file)}", unit="lines"):
                    try:
                        record = json.loads(line.strip())
                        if not isinstance(record, list):
                            record = record['conversations']
                        file_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {input_file}: {e}")
                        continue
            
            file_counts[os.path.basename(input_file)] = len(file_records)
            all_records.extend(file_records)
            print(f"Records from {os.path.basename(input_file)}: {len(file_records)}")
            
        except FileNotFoundError:
            print(f"File not found: {input_file}")
            continue
    
    print(f"\nTotal records collected: {len(all_records)}")
    print("Shuffling records...")
    random.shuffle(all_records)
    
    # Write all records to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in tqdm(all_records, desc="Writing shuffled merged file", unit="records"):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return len(all_records)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    input_files = [
        "/data/jianyuan/LLMreasoning/prm_datasets/merged_training_dataset_processed.jsonl",
        "/data/jianyuan/LLMreasoning/prm_datasets/math_shepherd_processed.jsonl",
        "/data/jianyuan/LLMreasoning/prm_datasets/RLHFlow_DS-and-Mistral-PRM-Data.jsonl"
    ]
    
    output_file = "/data/jianyuan/LLMreasoning/prm_datasets/merged_data_unfiltered_v0.jsonl"
    
    total_rows = merge_jsonl_files(input_files, output_file)
    print(f"\nMerged data saved to: {output_file}")
    print(f"Final number of rows in merged file: {total_rows}")
