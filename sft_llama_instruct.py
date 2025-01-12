import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig
import deepspeed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train_dataset = load_math_data("math_reasoning_sft_v0.json")
train_dataset = load_dataset("O1-OPEN/OpenO1-SFT-Pro", split="train")


model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 


def preprocess_function(examples):
    inputs = []
    max_length = 4096  # Define the maximum length for each chunk
    for query, response in zip(examples["prompt"], examples["response"]):
        messages = [
            {"role": "system", "content": "You are a math expert. Try your best to solve the problem step by step."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
        templated_messages = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs.append(templated_messages)

        
    inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=4096, return_tensors="pt")
    # Prepare labels by shifting the input_ids
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)


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
    output_dir="./llama-sft-math-pro-2e",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
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
    model.to(device)
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
    model.save_pretrained("./llama-sft-math-pro-2e-full-merged")
    tokenizer.save_pretrained("./llama-sft-math-pro-2e-full-merged")
    print("Full model and tokenizer saved!")


if __name__ == "__main__":
    main()
