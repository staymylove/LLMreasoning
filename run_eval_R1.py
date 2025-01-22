import argparse
import numpy as np
import os
import json
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset
import re
from openai import OpenAI
import multiprocessing as mp
from tqdm import tqdm
import time

def extract_answer(solution_text: str):
    try:
        # 确保 solution_text 是字符串类型
        if not isinstance(solution_text, str):
            solution_text = str(solution_text)
        
        # 定义用于提取 boxed 内容的正则表达式
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        # 使用 re.findall 查找所有符合条件的匹配
        matches = re.findall(boxed_pattern, solution_text)
        
        # 如果找到匹配项，则返回最后一个匹配项的内容
        if matches:
            return matches[-1].strip()
        
        # 如果没有匹配项，则返回 None
        return None
    except Exception as e:
        # 捕获任何异常并打印错误信息
        print(f"Error occurred: {e}")
        return None

def apply_chat_template(toker, messages):
    input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return input_prompt

def prepare_input_boxed(template, input_d):
    problem = input_d['problem']
    steps = input_d['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_response = tagged_response.strip()
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

def save_generated_data(output_file, data):
    with open(output_file, 'a') as f:
        f.write(json.dumps(data) + '\n')

def single_process(args):

    
    d, template, model_name, use_voting, voting_n, output_dir = args
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key="sk-81ae6048413b40f7967e4cecacd2a6a5",
    )
    
    messages = prepare_input_boxed(template, d)
    
    if not use_voting:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
        )
        generated_critique = completion.choices[0].message.content
        pred = extract_answer(generated_critique)
        try:
            pred = int(pred)
        except:
            pred = None
    else:
        if 'Qwen2.5-Math' in model_name:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=8192,
                n=voting_n,
            )
        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1.0,
                top_p=0.9,
                max_tokens=8192,
                n=voting_n,
            )
        generated_critique = [choice.message.content for choice in completion.choices]
        preds = [extract_answer(e) for e in generated_critique]
        preds = [e for e in preds if e is not None]
        if len(preds) == 0:
            pred = None
        else:
            pred = Counter(preds).most_common(1)[0][0]
            try:
                pred = int(pred)
            except:
                pred = None
                
    result = {
        'generated_critique': generated_critique,
        'prediction': pred,
        'match': (pred == d['label'])
    }

# Save the result after generating
    save_generated_data(os.path.join(output_dir, 'generated_data.jsonl'), result)

    return result




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        choices=['gsm8k', 'math', 'olympiadbench', 'omnimath'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument('--use_voting', action='store_true')
    parser.add_argument('--voting_n', type=int, default=8)
    args = parser.parse_args()

    args.model_name = os.path.basename(args.model_path)
    TEMPLATE = open('./templates/critique_template.txt').read().strip()

    output_dir = os.path.join(args.output_dir, args.model_name if not args.use_voting else f'{args.model_name}_voting')
    os.makedirs(output_dir, exist_ok=True)

    # Load the specified data file
    if args.configs is None:
        args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

    for config in args.configs:
        if not args.use_voting:
            output_dir = os.path.join(args.output_dir, args.model_name)
        else:
            output_dir = os.path.join(args.output_dir, f'{args.model_name}_voting')
        os.makedirs(output_dir, exist_ok=True)

        input_data = load_dataset('Qwen/ProcessBench', split=config)

    process_args = [(d, TEMPLATE, args.model_name, args.use_voting, args.voting_n, output_dir) for d in input_data]
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(single_process, process_args),
                          total=len(process_args), desc='Processing data'))

    res_data = []
    for i, d in enumerate(input_data):
        new_d = d.copy()
        new_d.update(results[i])
        res_data.append(new_d)

    error_data = [e for e in res_data if e['label'] != -1]
    correct_data = [e for e in res_data if e['label'] == -1]

    with open(os.path.join(output_dir, 'error.jsonl'), 'w') as f:
        for e in error_data:
            f.write(json.dumps(e) + '\n')
    with open(os.path.join(output_dir, 'correct.jsonl'), 'w') as f:
        for e in correct_data:
            f.write(json.dumps(e) + '\n')
    
    acc1 = np.mean([e['match'] for e in error_data]) * 100
    acc2 = np.mean([e['match'] for e in correct_data]) * 100
    f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    print(f'Error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

if __name__ == '__main__':
    main()
