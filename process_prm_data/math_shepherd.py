import json
from tqdm import tqdm


with open('/data/jianyuan/LLMreasoning/prm_datasets/peiyi9979_Math-Shepherd.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

step_tag2 = 'ки'
examples = []
for data in tqdm(dataset):
    # the first sentence is the question
    question = ''

    # the rest in the input is the process
    process = data['input']

    # the label is the last sentence
    label_indices = [i for i, label in enumerate(data['input'].split()) if label == step_tag2]
    labels = [s for i, s in enumerate(data['label'].split()) if i in label_indices]

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
with open('/data/jianyuan/LLMreasoning/prm_datasets/math_shepherd_processed.jsonl', 'w') as f:
    for example in processed_examples:
        f.write(json.dumps(example) + '\n')
