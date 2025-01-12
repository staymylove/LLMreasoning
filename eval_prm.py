from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import os

from peft import PeftModel,PeftConfig
from peft import get_peft_model, LoraConfig, TaskType
# Ensure bitsandbytes is available for 8-bit quantization
# import bitsandbytes as bnb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from torch.nn import BCEWithLogitsLoss
from transformers import DataCollatorWithPadding
from datasets import concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-4)

args = parser.parse_args()


good_token = '+'
bad_token = '-'
step_tag = '\n\n' #ки
step_tag2 = '\n\n'

model_path = "/data/zeju/llama"

# tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    add_eos_token=False, 
)

print(tokenizer.encode('a ки b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a \n\n b'))
print(tokenizer.encode('a b'))
print(tokenizer.encode('a \n\n\n b'))
print(tokenizer.encode('a b'))


print(tokenizer.encode('a \n\n\n\n b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a \n\n\n\n\n b'))
print(tokenizer.encode('a b'))


print(tokenizer.encode('a + b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a - b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode(' + -'))
print(tokenizer.encode('+-'))


# if USE_8bit is True:
#     model = prepare_model_for_int8_training(model)
print(tokenizer.eos_token_id)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference


# tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}") # [488, 481]
print(candidate_tokens)
step_tag_id = tokenizer.encode(f" {step_tag2}")[-1] # 76325
print('step_tag_id:',tokenizer.encode(f" {step_tag}"))
print('step_tag_id2:',tokenizer.encode(f"{step_tag2}"))
# model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()
# model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # load_in_8bit=True,   # Enables 8-bit quantization
    # device_map="auto",   # Automatically assigns the model to available GPUs/CPUs
    # torch_dtype=torch.float16,  # Mixed precision for faster inference
    # attn_implementation="flash_attention_2",
)


adapter_config = PeftConfig.from_pretrained('/data/zeju/LLMreasoning/fine_tuned_llama_1b_mix_lora_16bit1')

# Wrap the pre-trained model with the LoRA fine-tuned weights
model = PeftModel.from_pretrained(model, '/data/zeju/LLMreasoning/fine_tuned_llama_1b_mix_lora_16bit1')

# model.to('cuda:0')
# print(model.device)
question = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
output1 = "Step 1: Janet's ducks lay 16 eggs per day. \n\n Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. \n\n Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. \n\n Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 \n\n" # 18 is right
output2 = "Step 1: Janet's ducks lay 16 eggs per day. \n\n Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. \n\n Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. \n\n Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 \n\n" # 17 is wrong


# question = "A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction."

# output1= "<Thought>\nAlright, I need to find the probability of the spinner landing on C. We're given the probabilities for A and B, which are 1/3 and 5/12 respectively. The spinner is divided into three parts, so the sum of the probabilities of A, B, and C must equal 1. \n\nFirst, I'll add the given probabilities: \n\n1/3 + 5/12 \n\nTo add these fractions, I need to find a common denominator. The least common multiple of 3 and 12 is 12. So, I'll convert 1/3 to have a denominator of 12: \n\n1/3 = 4/12 \n\nNow, I can add: \n\n4/12 + 5/12 = 9/12 \n\nThe probability of A or B is 9/12. To find the probability of C, I'll subtract this from 1: \n\n1 - 9/12 \n\nTo subtract a fraction from 1, I can convert 1 to 12/12 and then subtract: \n\n12/12 - 9/12 = 3/12 \n\nSimplifying 3/12 gives me 1/4. \n\nSo, the probability of the spinner landing on C is 1/4.\n</Thought> \n\n<Output>\n\\boxed{\\frac{1}{4}}\n</Output> \n\n"


# output2= "<Thought>\nAlright, I'm tasked with finding the probability of a board game spinner landing on part \\( C \\), given that the spinner is divided into three parts labeled \\( A \\), \\( B \\), and \\( C \\). The probabilities of landing on \\( A \\) and \\( B \\) are provided as \\( \\frac{1}{3} \\) and \\( \\frac{5}{12} \\) respectively. \n\nFirst, let's summarize what I know: \n\n1. **Probability of landing on \\( A \\)**: \\( \\frac{1}{3} \\)\n2. **Probability of landing on \\( B \\)**: \\( \\frac{5}{12} \\)\n3. **Total parts on the spinner**: 3 (since it's divided into three parts) \n\nThe total probability of landing on any part of the spinner must add up to 1, since one of the three parts must always land on. Therefore, the probability of landing on \\( C \\) can be calculated as: \n\n\\[ P(C) = 1 - P(A) - P(B) \\] \n\nSubstituting the given probabilities: \n\n\\[ P(C) = 1 - \\frac{1}{3} - \\frac{5}{12} \\] \n\nTo simplify this, I need to find a common denominator. The least common multiple of 3 and 12 is 12, so I'll convert \\( \\frac{1}{3} \\) to have a denominator of 12: \n\n\\[ \\frac{1}{3} = \\frac{4}{12} \\] \n\nNow, I can substitute this back into the equation: \n\n\\[ P(C) = 1 - \\frac{4}{12} - \\frac{5}{12} \\] \n\nCombining the fractions: \n\n\\[ P(C) = 1 - \\frac{9}{12} \\] \n\nSince \\( 1 \\) can be written as \\( \\frac{12}{12} \\), the equation becomes: \n\n\\[ P(C) = \\frac{12}{12} - \\frac{9}{12} \\] \n\nSimplifying: \n\n\\[ P(C) = \\frac{3}{12} \\] \n\nFurther simplifying by dividing both the numerator and the denominator by their greatest common divisor, which is 3: \n\n\\[ P(C) = \\frac{1}{4} \\] \n\nTherefore, the probability of the spinner landing on \\( C \\) is \\( \\frac{1}{4} \\).\n</Thought> \n\n<Output>\n\\(\\frac{1}{4}\\)\n</Output> \n\n"
 
 # label= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  "label": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


for output in [output1,output2]:
# for output in [output1, output2,output3]:
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])
    # print(input_id)

    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        # print(logits)
        scores = logits.softmax(dim=-1)[:,:,0] 
        # print(scores)
        step_scores = scores[input_id == step_tag_id]
        
        print(step_scores)
        print('aaaaaa')        
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])

