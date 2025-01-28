import json
import os
import random
from typing import List
import math
import argparse
from datasets import load_dataset

def sample_questions(input_filepath: str, output_filepath: str, num_samples: int):
    """
    Samples a specified number of questions from the input JSON file and saves them to an output JSON file.

    Parameters:
    - input_filepath (str): Path to the original JSON file containing all questions.
    - output_filepath (str): Path to save the sampled questions JSON file.
    - num_samples (int): Number of questions to sample.
    """
    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    # Sample questions
    sampled_questions = random.sample(questions, min(num_samples, len(questions)))

    # Save sampled questions to the output file
    with open(output_filepath, 'w') as f:
        json.dump(sampled_questions, f, indent=4)

    print(f"Saved {len(sampled_questions)} sampled questions to {output_filepath}")


def split_questions(input_filepath: str, output_dir: str, questions_per_file: int):
    """
    Splits the input JSON file into multiple smaller JSON files, each containing a specified number of questions.

    Parameters:
    - input_filepath (str): Path to the original JSON file containing all questions.
    - output_dir (str): Directory to save the split JSON files.
    - questions_per_file (int): Number of questions per split file.
    """
    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Split questions into chunks and save each chunk as a separate file
    for i in range(0, len(questions), questions_per_file):
        chunk = questions[i:i + questions_per_file]
        output_filepath = os.path.join(output_dir, f"questions_part_{i // questions_per_file + 1}.json")
        with open(output_filepath, 'w') as f:
            json.dump(chunk, f, indent=4)

        print(f"Saved {len(chunk)} questions to {output_filepath}")


def split_questions_uniformly(input_filepath: str, output_directory: str, num_files: int):
    """
    Split a JSON file containing questions into a specified number of files with approximately equal questions.

    Parameters:
    - input_filepath (str): Path to the JSON file containing the list of questions.
    - output_directory (str): Directory to save the split JSON files.
    - num_files (int): Number of files to split the questions into.

    Each output file will contain approximately len(questions) / num_files questions.
    """
    # Load questions from the input file
    with open(input_filepath, 'r') as f:
        questions = json.load(f)

    # Calculate the number of questions per file
    total_questions = len(questions)
    questions_per_file = math.ceil(total_questions / num_files)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Split the questions and write to output files
    for i in range(num_files):
        start_idx = i * questions_per_file
        end_idx = min(start_idx + questions_per_file, total_questions)
        questions_subset = questions[start_idx:end_idx]

        output_filepath = os.path.join(output_directory, f"questions_part_{i + 1}.json")
        with open(output_filepath, 'w') as f_out:
            json.dump(questions_subset, f_out, indent=4)

        print(f"Saved {len(questions_subset)} questions to {output_filepath}")


def split_dataset_kfold(dataset_name: str, output_dir: str, k_folds: int):
    """
    Loads a dataset from HuggingFace and splits it into k folds.
    Only extracts 'query' and 'answer' fields and renames them to 'problem' and 'final_answer'.

    Parameters:
    - dataset_name (str): Name of the HuggingFace dataset
    - output_dir (str): Directory to save the split JSON files
    - k_folds (int): Number of folds to split the dataset into
    """
    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name)
    
    # Convert dataset to list and extract only required fields with new names
    questions = []
    for item in dataset['train']:
        questions.append({
            'problem': item['query'],
            'final_answer': item['answer']
        })
    
    # Calculate the size of each fold
    total_questions = len(questions)
    fold_size = math.ceil(total_questions / k_folds)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the questions and write to output files
    for i in range(k_folds):
        start_idx = i * fold_size
        end_idx = min(start_idx + fold_size, total_questions)
        fold_questions = questions[start_idx:end_idx]
        
        output_filepath = os.path.join(output_dir, f"fold_{i + 1}.json")
        with open(output_filepath, 'w') as f_out:
            json.dump(fold_questions, f_out, indent=4)
            
        print(f"Saved {len(fold_questions)} questions to {output_filepath}")


def main():
    parser = argparse.ArgumentParser(description='Split HuggingFace dataset into k folds')
    parser.add_argument('--dataset', type=str, default='zeju-0727/O1_filter',
                        help='HuggingFace dataset name (default: zeju-0727/O1_filter)')
    parser.add_argument('--output_dir', type=str, default='output_folds_8',
                        help='Output directory for the fold files')
    parser.add_argument('--k_folds', type=int, default=8,
                        help='Number of folds to split the dataset into (default: 8)')
    
    args = parser.parse_args()
    
    split_dataset_kfold(args.dataset, args.output_dir, args.k_folds)


# Example usage
if __name__ == "__main__":
    main()

# Sampling a subset of questions from the original file
#   sample_questions("extracted_problems_and_answers.json", "sampled_questions.json", 10)

# Splitting the original file into multiple files with each containing 5 questions
#   split_questions("extracted_problems_and_answers.json", "output_directory", 5)
