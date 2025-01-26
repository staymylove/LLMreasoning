import json
from tqdm import tqdm
import re


with open('/data/jianyuan/LLMreasoning/prm_datasets/merged_training_dataset.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

step_tag2 = '\n\n'
examples = []
for data in tqdm(dataset):
    # the first sentence is the question
    question = data['question']

    # the rest in the input is the process
    process = data['process'].replace('<Thought>', '').replace('</Thought>', '')
    process = process.replace('<Output>', '').replace('</Output>', '')
    process = re.sub(r'\n{3,}', '\n\n', process)

    # the label is the last sentence
    labels = ['+' if l == 1 else '-' for l in data['label']]

    examples.append({'question': question, 'process': process, 'label': labels})
    

processed_examples = []
for example in examples:
    input_text = " ".join([example['question'], example['process']])
    steps = input_text.split(step_tag2)
    steps = [step.strip() for step in steps]
    labels = example['label']
    
    if len(steps) != len(labels):
        labels = labels[:len(steps)]
    
    massages = []
    for step, label in zip(steps, labels):
        massages.append({'role': 'user', 'content': step})
        massages.append({'role': 'assistant', 'content': label})
        
    processed_examples.append(massages)
    

# save processed examples to jsonl
with open('/data/jianyuan/LLMreasoning/prm_datasets/merged_training_dataset_processed.jsonl', 'w') as f:
    for example in processed_examples:
        f.write(json.dumps(example) + '\n')
