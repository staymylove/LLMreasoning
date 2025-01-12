# LLMreasoning

# visualize the generated data

/data/zeju/LLMreasoning/visualize_tree.ipynb

# Build our dataset
python /data/zeju/LLMreasoning/dataset.py
python /data/zeju/LLMreasoning/data_prepare.py


# Fine-tune LLama to get PRM
python /data/zeju/LLMreasoning/finetune_llama.py

# Evaluate LLama-PRM
python /data/zeju/LLMreasoning/eval_prm.py
