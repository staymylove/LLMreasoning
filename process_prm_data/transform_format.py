import json
import os
from tqdm import tqdm

def transform_data_format(input_file, output_file):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    transformed_records = []
    
    # Count lines first for the progress bar
    with open(input_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    
    # Read and transform records
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=num_lines, desc="Transforming records")):
            try:
                messages = json.loads(line.strip())
                
                # Skip if not a list of messages
                if not isinstance(messages, list):
                    messages = messages['conversations']
                
                # Extract problem and steps
                problem = None
                steps = []
                first_incorrect_step = None
                final_answer_correct = True
                
                for i, msg in enumerate(messages):
                    if msg.get('role', '').lower() == 'user':
                        content = msg.get('content', '').strip()
                        if not problem:
                            problem = content.split('?')[0] + '?'
                            steps.append(content.split('?')[-1])
                        else:
                            steps.append(content)
                    else:
                        # Check if this step is incorrect
                        if '-' in msg.get('content', ''):
                            final_answer_correct = False
                            if first_incorrect_step is None:
                                # The step index is (i-1)//2 since we have alternating user/assistant messages
                                first_incorrect_step = len(steps) - 1
                
                if not problem or not steps:
                    continue
                
                # Create new format
                new_record = {
                    "id": f"prm-{idx}",
                    "generator": "math-shepherd",  # Since this is from math shepherd dataset
                    "problem": problem,
                    "steps": steps,
                    "final_answer_correct": final_answer_correct,
                    "label": first_incorrect_step if first_incorrect_step is not None else -1
                }
                
                transformed_records.append(new_record)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {idx}: {e}")
                continue
    
    # Write transformed records
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in tqdm(transformed_records, desc="Writing transformed data"):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return len(transformed_records)

if __name__ == "__main__":
    input_file = "/data/jianyuan/LLMreasoning/prm_datasets/merged_data_unfiltered_v0.jsonl"
    output_file = "/data/jianyuan/LLMreasoning/prm_datasets/merged_data_unfiltered_v0_transformed.jsonl"
    
    total_records = transform_data_format(input_file, output_file)
    print(f"\nTransformed data saved to: {output_file}")
    print(f"Total number of transformed records: {total_records}") 
    
    # split erged_data_unfiltered_v0_transformed.jsonl into 3 files
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(3):
        with open(f"merged_data_unfiltered_v0_transformed_split_{i}.jsonl", 'w', encoding='utf-8') as f:
            for line in lines[i::3]:
                f.write(line)
