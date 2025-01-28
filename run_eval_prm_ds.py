

import os
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI
from datasets import load_dataset

# Create a global client variable
client = None


def single_process(d):
    global client
    steps = d['steps']
    messages = []
    for sdx, step in enumerate(steps):
        if sdx == 0:
            messages.append({
                'role': 'user', 
                'content': f"Problem: {d['problem']}\n\nStep: {step}\n\nIs this step correct? Please respond and answer with '+' for correct or '-' for incorrect in the end."
            })
        else:
            messages.append({
                'role': 'user', 
                'content': f"Step: {step}\n\nIs this step correct? Please respond and answer with '+' for correct or '-' for incorrect in the end."
            })
        
        completion = client.chat.completions.create(
            model='DS14B',
            messages=messages,
            n=1,
            temperature=0.,
            max_tokens=1,
        )
        print(completion)
        judgment = completion.choices[0].message.content.strip().lower().startswith('+')
        if not judgment:
            return sdx
        messages.append({'role': 'assistant', 'content': '+'})
    return -1

def main():
    global client
    # Initialize the client
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    os.makedirs('outputs/DS14B', exist_ok=True)

    configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    for config in configs:
        input_data = load_dataset('Qwen/ProcessBench', split=config)
        with Pool(32) as p:
            predictions = list(tqdm(p.imap(single_process, input_data), total=len(input_data),
                                    desc=f'Processing {config}', dynamic_ncols=True))
        
        res_data = []
        for idx, d in enumerate(input_data):
            new_d = d.copy()
            new_d['prediction'] = predictions[idx]
            new_d['match'] = predictions[idx] == d['label']
            res_data.append(new_d)
        
        data1 = [e for e in res_data if e['label'] != -1]
        data2 = [e for e in res_data if e['label'] == -1]
        with open(f'outputs/DS14B/{config}_error.jsonl', 'w') as f:
            for e in data1:
                f.write(json.dumps(e) + '\n')
        with open(f'outputs/DS14B/{config}_correct.jsonl', 'w') as f:
            for e in data2:
                f.write(json.dumps(e) + '\n')
        
        acc1 = np.mean([e['match'] for e in data1]) * 100
        acc2 = np.mean([e['match'] for e in data2]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

if __name__ == '__main__':
    main()
