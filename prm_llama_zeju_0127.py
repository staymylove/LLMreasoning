import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

processed_dataset_path = "/obs-gczx-project03/unfiltered-train/merged_data_unfiltered_v0.arrow"

if os.path.exists(processed_dataset_path):
    train_dataset = Dataset.load_from_disk(processed_dataset_path)
else:
    step_tag2 = 'ки'
    with open("/root/filter_data/balanced_data.jsonl", "r") as f:
        data_list = []
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            messages = json.loads(line)
            try:
                if not isinstance(messages, list):
                    messages = messages['conversations']
                inputs = tokenizer.apply_chat_template(messages, tokenize=False)
                labels = [
                    msg if msg['role'] == 'user' else {"role": "assistant", "content": step_tag2} for msg in messages
                ]
                labels = tokenizer.apply_chat_template(labels, tokenize=False)
                data_list.append(
                    {
                        'inputs': inputs,
                        'labels': labels,
                    }
                )
            except:
                print(messages)

    train_dataset = Dataset.from_list(data_list)
    
    good_token = '+'
    bad_token = '-'
    step_tag2 = 'ки'
    candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}")[1:] # [489, 482]
    step_tag_id = tokenizer.encode(f"{step_tag2}")[-1] # 77425
    print(f"Token IDs being used for ghost attention: {candidate_tokens}")
    print("step_tag_id: ", step_tag_id, tokenizer.encode(f" {step_tag2}")[-1])# For verification

    def preprocess_function(examples):
        max_length = 512  # Define the maximum length for each chunk
        inputs = examples['inputs']
        labels = examples['labels']
        inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        labels = tokenizer(labels, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        
        # implement the ghost attention
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][labels["input_ids"] != step_tag_id] = -100
        return inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
    train_dataset.save_to_disk(processed_dataset_path)


# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# model = AutoModelForCausalLM.from_pretrained(model_id)
# model.to(device)
lora_config = LoraConfig(
    r=8,  # Low-rank approximation dimension
    lora_alpha=16,  # Scaling factor for LoRA layers
    lora_dropout=0.1,  # Dropout for LoRA layers
    bias="none",  # Bias configuration (you can set it to "all" or "none" based on your needs)
)

# model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="llama_8b_filterd_data",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training if supported
    dataloader_num_workers=4,
    deepspeed="./ds_config.json"  # Path to DeepSpeed config file
)


def main():
    # Load and prepare the model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # model.to(device)
    model = get_peft_model(model, lora_config)

    # Prepare the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Merge the LoRA adapters into the model
    model = model.merge_and_unload()
    # Save the full model with merged adapters
    model.save_pretrained("./llama_8b_filterd_data-ft")
    tokenizer.save_pretrained("./llama_8b_filterd_data-ft")
    print("Full model and tokenizer saved!")


if __name__ == "__main__":
    main()
