from datasets import load_dataset
import json
import os
from tqdm import tqdm

def download_and_save_dataset(dataset_name, output_dir, split="train"):
    """
    Download dataset from Hugging Face and save it to a JSONL file
    """
    print(f"Downloading {dataset_name}...")
    # try:
    dataset = load_dataset(dataset_name, split=split)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSONL
    output_file = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"Saved {len(dataset)} rows to {output_file}")
    return dataset
        
    # except Exception as e:
    #     print(f"Error downloading {dataset_name}: {str(e)}")
    #     return None

def main():
    # Define output directory
    output_dir = "prm_datasets"
    
    # List of datasets to download
    datasets = [
        # "RLHFlow/Mistral-PRM-Data",
        # "peiyi9979/Math-Shepherd",
        # "KbsdJames/MathMinos-Natural-language-feedback",
        # "openreasoner/MATH-APS",
        # "RLHFlow/DS-and-Mistral-PRM-Data",
        "alpayariyak/prm800k"
    ]
    
    # Download and save each dataset
    total_rows = 0
    for dataset_name in tqdm(datasets, desc="Downloading datasets"):
        dataset = download_and_save_dataset(dataset_name, output_dir)
        if dataset is not None:
            total_rows += len(dataset)
    
    print(f"\nDownloaded {len(datasets)} datasets with a total of {total_rows} rows")

if __name__ == "__main__":
    main()
