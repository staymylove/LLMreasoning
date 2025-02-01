from datasets import load_dataset
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

# Initialize vLLM and tokenizer
model_name = "/root/DeepSeek-R1-Distill-Qwen-32B"  # or your local model path
llm = LLM(
    model=model_name,
    tensor_parallel_size=8,  # adjust based on your GPU setup
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    stop=None
)

# Load the dataset from Hugging Face hub
dataset = load_dataset("zeju-0727/filter_data", split='train')

def is_all_correct(labels):
    """Check if all steps in an example are correct."""
    return all(label['content'] == "+" for label in labels if label['role'] == 'assistant')

def apply_chat_template(messages):
    """Apply the model's chat template to format messages."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(prompt, add_special_tokens=False).input_ids

def prepare_correct_messages(example):
    """Prepare messages for correct examples."""
    labels = example['labels']
    user_messages = []
    assistant_messages = []

    for i, label in enumerate(labels):
        if label['role'] == 'user':
            user_messages.append(label['content'])
        elif label['role'] == 'assistant':
            assistant_messages.append(label['content'])

    messages = []
    for i, label in enumerate(assistant_messages):
        step = user_messages[i]
        prompt = f"Step: {step}\n\nIs this step correct? You must answer with '+' for correct or '-' for incorrect in the end of your response."
        messages.append({
            'role': 'user',
            'content': prompt
        })
        if label == "+":
            messages.append({
                'role': 'assistant',
                'content': "<think>\n\n</think> +"
            })
    return messages

def prepare_incorrect_messages(examples):
    """Prepare all prompts for incorrect examples in a single list."""
    all_negative_prompts = []
    all_negative_indices = []
    all_messages = []

    for example in tqdm(examples, desc="Preparing incorrect messages"):
        labels = example['labels']
        user_messages = []
        assistant_messages = []

        for i, label in enumerate(labels):
            if label['role'] == 'user':
                user_messages.append(label['content'])
            elif label['role'] == 'assistant':
                assistant_messages.append(label['content'])

        messages = []
        negative_prompts = []
        negative_indices = []

        for i, label in enumerate(assistant_messages):
            step = user_messages[i]
            prompt = f"Step: {step}\n\nIs this step correct? You must answer with '+' for correct or '-' for incorrect in the end of your response."
            messages.append({
                'role': 'user',
                'content': prompt
            })
            if label == "+":
                messages.append({
                    'role': 'assistant',
                    'content': "<think>\n\n</think> +"
                })
            else:
                # For incorrect steps, collect the prompt for batch processing
                negative_prompts.append(apply_chat_template(messages.copy()))
                negative_indices.append(len(messages))
                # Add a placeholder that will be replaced later
                messages.append({
                    'role': 'assistant',
                    'content': None
                })
                break

        if negative_prompts:  # Only include examples with negative prompts
            all_messages.append(messages)
            all_negative_prompts.extend(negative_prompts)
            all_negative_indices.append((len(all_messages)-1, negative_indices))

    return all_messages, all_negative_prompts, all_negative_indices

def process_examples(examples):
    """Process multiple examples in batch."""
    all_messages, all_negative_prompts, all_negative_indices = prepare_incorrect_messages(examples)

    # Batch process all negative prompts
    if all_negative_prompts:
        outputs = llm.generate(prompt_token_ids=all_negative_prompts, sampling_params=sampling_params)

        # Map outputs back to their original messages
        current_output = 0
        for example_idx, indices in all_negative_indices:
            for idx in indices:
                generated_text = outputs[current_output].outputs[0].text
                # Check if '-' appears in last 5 characters using regex
                pattern = r'(?:\b\w+\b\s*){0,5}-$'
                if re.search(pattern, generated_text.strip().lower()):
                    all_messages[example_idx][idx]['content'] = generated_text
                current_output += 1

    return all_messages

if __name__ == '__main__':
    correct_examples = []
    incorrect_examples = []

    # First pass: categorize examples
    print("Categorizing examples...")
    for example in tqdm(dataset, desc="Categorizing"):
        if is_all_correct(example['labels']):
            correct_examples.append(example)
        else:
            incorrect_examples.append(example)

    print(f"Found {len(correct_examples)} examples with all correct steps")
    print(f"Found {len(incorrect_examples)} examples with some incorrect steps")

    results = []

    # Process all incorrect examples at once
    print("\nProcessing incorrect examples...")
    if incorrect_examples:
        batch_results = process_examples(incorrect_examples)
        results.extend(batch_results)

    print(f"\nProcessed {len(results)} total examples")

    # Save results to jsonl file
    output_file = "cot_data.jsonl"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        for messages in results:
            f.write(json.dumps({"conversations": messages}) + "\n")
    print("Done!")
