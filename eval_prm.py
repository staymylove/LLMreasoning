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
parser.add_argument("--learning_rate", type=float, default=2e-5)

args = parser.parse_args()


good_token = '+'
bad_token = '-'
step_tag = '\n\n' #ки
# step_tag2 = '\n\n'
step_tag2 = 'ки'

model_path = "meta-llama/Llama-3.1-8B-Instruct"

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
candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}")[1:] # [488, 481]
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


adapter_config = PeftConfig.from_pretrained('/data/jianyuan/LLMreasoning/fine_tuned_llama_math_shepherd_full')

# Wrap the pre-trained model with the LoRA fine-tuned weights
model = PeftModel.from_pretrained(model, '/data/jianyuan/LLMreasoning/fine_tuned_llama_math_shepherd_full')

# model.to('cuda:0')
# print(model.device)
question = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
output1 = "Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки" # 18 is right
output2 = "Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки" # 17 is wrong

# question = "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?"

# output2 = "Step 1: Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons. ки\nStep 2: She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week. ки\nStep 3: She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week. ки\nStep 4: Janet spends 120 + 140 = <<120+140=260>>260 on music lessons per week. ки\nStep 5: She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year. The answer is: 13520 ки"
# output1 = "Step 1: Janet pays $40/hour for 3 hours per week of clarinet lessons. ки\nStep 2: She spends 40 * 3 = $120 per week on clarinet lessons. ки\nStep 3: She spends 28 * 5 = $140 per week on piano lessons. ки\nStep 4: The difference in cost between piano and clarinet lessons is 140 - 120 = $20 per week. ки\nStep 5: There are 52 weeks in a year, so she spends 20 * 52 = $1040 more on piano lessons than clarinet lessons in a year. The answer is: 1040 ки" # 1040 is right


# question = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
# output1 = "Step 1: Janet's ducks lay 16 eggs per day. \n\n Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. \n\n Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. \n\n Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 \n\n" # 18 is right
# output2 = "Step 1: Janet's ducks lay 16 eggs per day. \n\n Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. \n\n Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. \n\n Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 \n\n" # 17 is wrong


# question = "A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction."

# output1= "<Thought>\nAlright, I need to find the probability of the spinner landing on C. We're given the probabilities for A and B, which are 1/3 and 5/12 respectively. The spinner is divided into three parts, so the sum of the probabilities of A, B, and C must equal 1. \n\nFirst, I'll add the given probabilities: \n\n1/3 + 5/12 \n\nTo add these fractions, I need to find a common denominator. The least common multiple of 3 and 12 is 12. So, I'll convert 1/3 to have a denominator of 12: \n\n1/3 = 4/12 \n\nNow, I can add: \n\n4/12 + 5/12 = 9/12 \n\nThe probability of A or B is 9/12. To find the probability of C, I'll subtract this from 1: \n\n1 - 9/12 \n\nTo subtract a fraction from 1, I can convert 1 to 12/12 and then subtract: \n\n12/12 - 9/12 = 3/12 \n\nSimplifying 3/12 gives me 1/4. \n\nSo, the probability of the spinner landing on C is 1/4.\n</Thought> \n\n<Output>\n\\boxed{\\frac{1}{4}}\n</Output> \n\n"


# output2= "<Thought>\nAlright, I'm tasked with finding the probability of a board game spinner landing on part \\( C \\), given that the spinner is divided into three parts labeled \\( A \\), \\( B \\), and \\( C \\). The probabilities of landing on \\( A \\) and \\( B \\) are provided as \\( \\frac{1}{3} \\) and \\( \\frac{5}{12} \\) respectively. \n\nFirst, let's summarize what I know: \n\n1. **Probability of landing on \\( A \\)**: \\( \\frac{1}{3} \\)\n2. **Probability of landing on \\( B \\)**: \\( \\frac{5}{12} \\)\n3. **Total parts on the spinner**: 3 (since it's divided into three parts) \n\nThe total probability of landing on any part of the spinner must add up to 1, since one of the three parts must always land on. Therefore, the probability of landing on \\( C \\) can be calculated as: \n\n\\[ P(C) = 1 - P(A) - P(B) \\] \n\nSubstituting the given probabilities: \n\n\\[ P(C) = 1 - \\frac{1}{3} - \\frac{5}{12} \\] \n\nTo simplify this, I need to find a common denominator. The least common multiple of 3 and 12 is 12, so I'll convert \\( \\frac{1}{3} \\) to have a denominator of 12: \n\n\\[ \\frac{1}{3} = \\frac{4}{12} \\] \n\nNow, I can substitute this back into the equation: \n\n\\[ P(C) = 1 - \\frac{4}{12} - \\frac{5}{12} \\] \n\nCombining the fractions: \n\n\\[ P(C) = 1 - \\frac{9}{12} \\] \n\nSince \\( 1 \\) can be written as \\( \\frac{12}{12} \\), the equation becomes: \n\n\\[ P(C) = \\frac{12}{12} - \\frac{9}{12} \\] \n\nSimplifying: \n\n\\[ P(C) = \\frac{3}{12} \\] \n\nFurther simplifying by dividing both the numerator and the denominator by their greatest common divisor, which is 3: \n\n\\[ P(C) = \\frac{1}{4} \\] \n\nTherefore, the probability of the spinner landing on \\( C \\) is \\( \\frac{1}{4} \\).\n</Thought> \n\n<Output>\n\\(\\frac{1}{4}\\)\n</Output> \n\n"
 
 # label= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  "label": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

# question = "Suppose $f(x)=\\frac{3}{2-x}$. If $g(x)=\\frac{1}{f^{-1}(x)}+9$, find $g(3)$."
# output1 = "<Thought>\nAlright, let's tackle this problem step by step. We're given two functions: \n\n1. \\( f(x) = \\frac{3}{2 - x} \\)\n2. \\( g(x) = \\frac{1}{f^{-1}(x)} + 9 \\) \n\nOur goal is to find \\( g(3) \\). To do this, we'll first need to find the inverse function \\( f^{-1}(x) \\) and then substitute \\( x = 3 \\) into \\( g(x) \\). \n\n**Step 1: Find the Inverse Function \\( f^{-1}(x) \\)** \n\nTo find the inverse of \\( f(x) = \\frac{3}{2 - x} \\), we swap \\( x \\) and \\( y \\) to get \\( x = \\frac{3}{2 - y} \\), and then solve for \\( y \\). \n\n\\[ x = \\frac{3}{2 - y} \\] \n\nMultiply both sides by \\( 2 - y \\) to clear the fraction: \n\n\\[ x(2 - y) = 3 \\] \n\nExpand the left side: \n\n\\[ 2x - xy = 3 \\] \n\nRearrange to isolate \\( y \\): \n\n\\[ xy = 2x - 3 \\] \n\nDivide both sides by \\( x \\) (assuming \\( x \\neq 0 \\)): \n\n\\[ y = \\frac{2x - 3}{x} \\] \n\nSimplify: \n\n\\[ y = 2 - \\frac{3}{x} \\] \n\nSo, the inverse function is: \n\n\\[ f^{-1}(x) = 2 - \\frac{3}{x} \\] \n\n**Step 2: Substitute \\( x = 3 \\) into \\( f^{-1}(x) \\)** \n\nNow, we need to find \\( f^{-1}(3) \\). \n\n\\[ f^{-1}(3) = 2 - \\frac{3}{3} \\] \n\nSimplify: \n\n\\[ f^{-1}(3) = 2 - 1 \\] \n\n\\[ f^{-1}(3) = 1 \\] \n\n**Step 3: Find \\( g(3) \\)** \n\nGiven \\( g(x) = \\frac{1}{f^{-1}(x)} + 9 \\), substitute \\( x = 3 \\) and \\( f^{-1}(3) = 1 \\): \n\n\\[ g(3) = \\frac{1}{f^{-1}(3)} + 9 \\] \n\n\\[ g(3) = \\frac{1}{1} + 9 \\] \n\n\\[ g(3) = 1 + 9 \\] \n\n\\[ g(3) = 10 \\] \n\nTherefore, \\( g(3) = 10 \\).\n</Thought> \n\n<Output>\n\\( g(3) = 10 \\)\n</Output> \n\n"

# output2 = "<Thought>\nAlright, I need to solve this problem step by step to ensure I get the correct answer. The problem involves two functions, \\( f(x) \\) and \\( g(x) \\), and we're asked to find the value of \\( g(3) \\). Let's start by understanding each function and then proceed to find \\( g(3) \\). \n\nFirst, let's analyze \\( f(x) \\):\n\\[ f(x) = \\frac{3}{2 - x} \\] \n\nThis is a rational function where the numerator is a constant and the denominator is a linear expression with a variable. To find the inverse of \\( f(x) \\), denoted as \\( f^{-1}(x) \\), I'll need to swap \\( x \\) and \\( y \\) and then solve for \\( y \\). \n\nSwapping \\( x \\) and \\( y \\):\n\\[ y = \\frac{3}{2 - x} \\]\nNow, I'll solve for \\( x \\) in terms of \\( y \\):\n\\[ x = 2 - \\frac{3}{y} \\]\nMultiplying both sides by \\( y \\) to get rid of the fraction:\n\\[ xy = 2y - 3 \\]\nMoving all terms to one side:\n\\[ xy - 2y = -3 \\]\nFactoring out \\( y \\):\n\\[ y(x - 2) = -3 \\]\nDividing both sides by \\( (x - 2) \\) to isolate \\( y \\):\n\\[ y = \\frac{-3}{x - 2} \\]\nSo, the inverse function \\( f^{-1}(x) \\) is:\n\\[ f^{-1}(x) = \\frac{-3}{x - 2} \\] \n\nNow, let's define \\( g(x) \\):\n\\[ g(x) = \\frac{1}{f^{-1}(x)} + 9 \\]\nSubstituting \\( f^{-1}(x) \\) into \\( g(x) \\):\n\\[ g(x) = \\frac{1}{\\frac{-3}{x - 2}} + 9 \\]\nSimplifying the fraction inside \\( g(x) \\):\n\\[ g(x) = \\frac{x - 2}{-3} + 9 \\]\n\\[ g(x) = -\\frac{x - 2}{3} + 9 \\]\n\\[ g(x) = -\\frac{x}{3} + \\frac{2}{3} + 9 \\]\n\\[ g(x) = -\\frac{x}{3} + \\frac{2 + 27}{3} \\]\n\\[ g(x) = -\\frac{x}{3} + \\frac{29}{3} \\] \n\n"


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

