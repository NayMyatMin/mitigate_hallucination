"""
Utility functions for data processing.
"""

import os
from typing import Dict, List, Union, Any, Optional
from datasets import load_dataset, Dataset, DatasetDict
import torch
from transformers import PreTrainedTokenizer
import random
import json
import numpy as np
import logging

from datasets import DownloadConfig
from config.data_config import DATASETS, PROMPT_TEMPLATES, DATA_PROCESSING

# Set up logging
logger = logging.getLogger(__name__)


def load_and_prepare_datasets(dataset_configs, max_samples=None):
    """
    Load and prepare datasets for evaluation.
    
    Args:
        dataset_configs: Dictionary of dataset configurations
        max_samples: Maximum number of samples to load per dataset split
        
    Returns:
        Dictionary of prepared datasets
    """
    datasets = {}
    
    # Load CoQA dataset
    if "coqa" in dataset_configs:
        try:
            print("Loading CoQA dataset...")
            # Load the CoQA dataset from HuggingFace
            coqa_dataset = load_dataset("coqa")
            
            # Process dataset to extract questions, answers, and contexts
            # Sample limit is set to match paper's 7983 QA pairs
            sample_limit = max_samples if max_samples else 8000  # Increased to match paper's 7983 QA pairs
            
            # For evaluation, use the validation split
            train_data = []
            eval_data = []
            
            # Process a small subset for training examples
            for i, sample in enumerate(coqa_dataset["train"]):
                if i >= 2:  # Just load a few for training
                    break
                
                # Process each question and answer pair in the CoQA sample
                story = sample["story"]
                questions = sample["questions"]
                answers = sample["answers"]
                
                # Check the structure by examining the first sample
                if i == 0:
                    print(f"CoQA sample structure:")
                    print(f"  - questions type: {type(questions)}, first element type: {type(questions[0]) if questions else 'N/A'}")
                    print(f"  - answers type: {type(answers)}, structure: {list(answers.keys()) if isinstance(answers, dict) else 'not a dict'}")
                
                # Handle different possible structures
                if isinstance(questions, list):
                    for j, q in enumerate(questions):
                        if j < len(answers.get("input_text", [])):
                            # Handle the case where questions is a list of strings and answers has input_text as a list
                            answer_text = answers["input_text"][j] if "input_text" in answers else ""
                            train_data.append({
                                "question": q,
                                "answer": answer_text,
                                "context": story
                            })
                        else:
                            break
                
            # Process validation split for evaluation examples
            for i, sample in enumerate(coqa_dataset["validation"]):
                if i >= sample_limit or i >= len(coqa_dataset["validation"]):
                    break
                
                # Process each question and answer pair in the CoQA sample
                story = sample["story"]
                questions = sample["questions"]
                answers = sample["answers"]
                
                # Print debug info for first validation sample
                if i == 0:
                    print(f"CoQA validation sample structure:")
                    print(f"  - questions type: {type(questions)}, length: {len(questions) if hasattr(questions, '__len__') else 'N/A'}")
                    print(f"  - answers type: {type(answers)}")
                
                # Handle different possible structures
                if isinstance(questions, list):
                    for j, q in enumerate(questions):
                        if j < len(answers.get("input_text", [])):
                            # Handle the case where questions is a list of strings and answers has input_text as a list
                            answer_text = answers["input_text"][j] if "input_text" in answers else ""
                            eval_data.append({
                                "question": q,
                                "answer": answer_text,
                                "context": story
                            })
                        else:
                            break
                        
                        # Limit total data size
                        if len(eval_data) >= sample_limit:
                            break
                
                if len(eval_data) >= sample_limit:
                    break
            
            # Create a dataset dictionary with train and eval splits
            datasets["coqa"] = {
                "train": Dataset.from_list(train_data),
                "eval": Dataset.from_list(eval_data)
            }
            
            print(f"Loaded CoQA dataset:")
            print(f"  - Train size: {len(datasets['coqa']['train'])}")
            print(f"  - Eval size: {len(datasets['coqa']['eval'])}")
            
            # Print dataset info
            for split in datasets["coqa"]:
                print(f"Dataset coqa, split {split}:")
                print(f"  - Type: {type(datasets['coqa'][split])}")
                print(f"  - Column names: {datasets['coqa'][split].column_names}")
                print(f"  - Size: {len(datasets['coqa'][split])}")
                print(f"  - First example: {datasets['coqa'][split][0]}")
            
        except Exception as e:
            print(f"Error loading CoQA dataset: {e}")
            # Create an empty dataset
            datasets["coqa"] = {
                "train": Dataset.from_list([]),
                "eval": Dataset.from_list([])
            }
    
    # Load SQuAD v2.0 dataset
    if "squad_v2" in dataset_configs:
        try:
            print("Loading SQuAD v2.0 dataset...")
            # Load the SQuAD v2.0 dataset from HuggingFace
            squad_dataset = load_dataset("squad_v2")
            
            # Print total validation samples
            print(f"SQuAD v2.0 total validation samples: {len(squad_dataset['validation'])}")
            
            # Determine which samples are actually answerable (non-impossible)
            answerable_samples = []
            for sample in squad_dataset["validation"]:
                if sample["answers"]["text"]:  # If there's at least one answer
                    answerable_samples.append(sample)
            
            print(f"SQuAD v2.0 answerable validation samples: {len(answerable_samples)}")
            
            # Set sample limit to match paper's 5928 QA pairs
            sample_limit = max_samples if max_samples else 6000  # Default is 6000, enough to cover paper's 5928
            
            # Function to process SQuAD samples
            def process_squad(sample):
                return {
                    "context": sample["context"],
                    "question": sample["question"],
                    "answer": sample["answers"]["text"][0] if sample["answers"]["text"] else ""
                }
            
            # Process for training (just a small subset)
            train_data = []
            for i, sample in enumerate(answerable_samples):
                if i >= 2:  # Just load a few for training examples
                    break
                train_data.append(process_squad(sample))
            
            # Process for evaluation (using the validation split)
            eval_data = []
            for i, sample in enumerate(answerable_samples):
                if i >= sample_limit:
                    break
                eval_data.append(process_squad(sample))
            
            # Create a dataset dictionary with train and eval splits
            datasets["squad_v2"] = {
                "train": Dataset.from_list(train_data),
                "eval": Dataset.from_list(eval_data)
            }
            
            print(f"Loaded SQuAD v2.0 dataset:")
            print(f"  - Train size: {len(datasets['squad_v2']['train'])}")
            print(f"  - Eval size: {len(datasets['squad_v2']['eval'])}")
            
            # Print dataset info
            for split in datasets["squad_v2"]:
                print(f"Dataset squad_v2, split {split}:")
                print(f"  - Type: {type(datasets['squad_v2'][split])}")
                print(f"  - Column names: {datasets['squad_v2'][split].column_names}")
                print(f"  - Size: {len(datasets['squad_v2'][split])}")
                print(f"  - First example: {datasets['squad_v2'][split][0]}")
            
        except Exception as e:
            print(f"Error loading SQuAD v2.0 dataset: {e}")
            # Create an empty dataset
            datasets["squad_v2"] = {
                "train": Dataset.from_list([]),
                "eval": Dataset.from_list([])
            }
    
    # Load TriviaQA dataset
    if "triviaqa" in dataset_configs:
        try:
            print("Loading TriviaQA dataset...")
            # Load the TriviaQA dataset from HuggingFace
            triviaqa_dataset = load_dataset("trivia_qa", "rc.nocontext")
            
            # Set sample limit to match paper's 9960 QA pairs
            sample_limit = max_samples if max_samples else 10000  # Increased to match paper's 9960 QA pairs
            
            # Function to process TriviaQA samples
            def process_triviaqa(sample):
                return {
                    "question": sample["question"],
                    "answer": sample["answer"]["value"],
                    "context": ""  # TriviaQA rc.nocontext has no context
                }
            
            # Process for training (just a small subset)
            train_data = []
            for i, sample in enumerate(triviaqa_dataset["train"]):
                if i >= 2:  # Just load a few for training examples
                    break
                train_data.append(process_triviaqa(sample))
            
            # Process for evaluation (using the validation split)
            eval_data = []
            for i, sample in enumerate(triviaqa_dataset["validation"]):
                if i >= sample_limit:
                    break
                eval_data.append(process_triviaqa(sample))
            
            # Create a dataset dictionary with train and eval splits
            datasets["triviaqa"] = {
                "train": Dataset.from_list(train_data),
                "eval": Dataset.from_list(eval_data)
            }
            
            print(f"Loaded TriviaQA dataset:")
            print(f"  - Train size: {len(datasets['triviaqa']['train'])}")
            print(f"  - Eval size: {len(datasets['triviaqa']['eval'])}")
            
            # Print dataset info
            for split in datasets["triviaqa"]:
                print(f"Dataset triviaqa, split {split}:")
                print(f"  - Type: {type(datasets['triviaqa'][split])}")
                print(f"  - Column names: {datasets['triviaqa'][split].column_names}")
                print(f"  - Size: {len(datasets['triviaqa'][split])}")
                print(f"  - First example: {datasets['triviaqa'][split][0]}")
            
        except Exception as e:
            print(f"Error loading TriviaQA dataset: {e}")
            # Create an empty dataset
            datasets["triviaqa"] = {
                "train": Dataset.from_list([]),
                "eval": Dataset.from_list([])
            }
    
    # Load HaluEval QA dataset
    if "halueval_qa" in dataset_configs:
        try:
            print("Loading HaluEval QA dataset...")
            # Load the HaluEval QA dataset from local file
            import json
            import os
            
            halueval_path = os.path.join("dataset", "HaluEval", "qa_data.json")
            if not os.path.exists(halueval_path):
                raise FileNotFoundError(f"HaluEval QA dataset file not found at: {halueval_path}")
                
            # Parse the jsonl file
            data = []
            with open(halueval_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            
            print(f"Loaded {len(data)} samples from HaluEval QA dataset")
            
            # Set sample limit (use all by default)
            sample_limit = max_samples if max_samples else len(data)
            
            # Function to process HaluEval QA samples
            def process_halueval_qa(sample):
                return {
                    "question": sample["question"],
                    "answer": sample["right_answer"],
                    "context": sample["knowledge"],
                    "hallucinated_answer": sample["hallucinated_answer"]
                }
            
            # Process for training (just a small subset)
            train_data = []
            for i, sample in enumerate(data[:2]):  # Just load a few for training examples
                train_data.append(process_halueval_qa(sample))
            
            # Process for evaluation
            eval_data = []
            for i, sample in enumerate(data):
                if i >= sample_limit:
                    break
                eval_data.append(process_halueval_qa(sample))
            
            # Create a dataset dictionary with train and eval splits
            datasets["halueval_qa"] = {
                "train": Dataset.from_list(train_data),
                "eval": Dataset.from_list(eval_data[:sample_limit])
            }
            
            print(f"Loaded HaluEval QA dataset:")
            print(f"  - Train size: {len(datasets['halueval_qa']['train'])}")
            print(f"  - Eval size: {len(datasets['halueval_qa']['eval'])}")
            print(f"  - Using hallucination evaluation: True")
            
            # Print dataset info
            for split in datasets["halueval_qa"]:
                print(f"Dataset halueval_qa, split {split}:")
                print(f"  - Type: {type(datasets['halueval_qa'][split])}")
                print(f"  - Column names: {datasets['halueval_qa'][split].column_names}")
                print(f"  - Size: {len(datasets['halueval_qa'][split])}")
                print(f"  - First example: {datasets['halueval_qa'][split][0]}")
            
        except Exception as e:
            print(f"Error loading HaluEval QA dataset: {e}")
            # Create an empty dataset
            datasets["halueval_qa"] = {
                "train": Dataset.from_list([]),
                "eval": Dataset.from_list([])
            }
    
    # Load TruthfulQA dataset
    if "truthfulqa" in dataset_configs:
        try:
            print("Loading TruthfulQA dataset...")
            
            # Try to load the local CSV file first (updated binary choice format)
            try:
                csv_path = os.path.join("dataset", "TruthfulQA.csv")
                print(f"Loading TruthfulQA from local CSV file: {csv_path}")
                csv_dataset = load_dataset("csv", data_files={"train": csv_path})
                
                # Print the CSV dataset info
                print(f"Loaded TruthfulQA from local CSV file")
                print(f"TruthfulQA dataset splits: {list(csv_dataset.keys())}")
                print(f"Train split features: {csv_dataset['train'].column_names}")
                print(f"First example keys: {list(csv_dataset['train'][0].keys())}")
                
                # Process the CSV dataset to create our binary choice format
                train_data = []
                eval_data = []
                
                # Process just a small subset for training
                for i, row in enumerate(csv_dataset["train"]):
                    if i >= 2:  # Just load a few for examples
                        break
                    
                    # Create binary choice format
                    example = {
                        "question": row["Question"],
                        "answer": row["Best Answer"],
                        "context": "",
                        "choices": [row["Best Answer"], row["Best Incorrect Answer"]],
                        "labels": [1, 0],
                        "is_binary_choice": True
                    }
                    train_data.append(example)
                
                # Process all for evaluation (or up to sample_limit)
                sample_limit = max_samples if max_samples else len(csv_dataset["train"])
                
                for i, row in enumerate(csv_dataset["train"]):
                    if i >= sample_limit:
                        break
                    
                    # Create binary choice format
                    example = {
                        "question": row["Question"],
                        "answer": row["Best Answer"],
                        "context": "",
                        "choices": [row["Best Answer"], row["Best Incorrect Answer"]],
                        "labels": [1, 0],
                        "is_binary_choice": True
                    }
                    eval_data.append(example)
                
                # Create a dataset dictionary with train and eval splits
                datasets["truthfulqa"] = {
                    "train": Dataset.from_list(train_data),
                    "eval": Dataset.from_list(eval_data)
                }
                
                print(f"Loaded TruthfulQA dataset:")
                print(f"  - Train size: {len(datasets['truthfulqa']['train'])}")
                print(f"  - Eval size: {len(datasets['truthfulqa']['eval'])}")
                print(f"  - Using binary choice format: True")
                
                # Print dataset info
                for split in datasets["truthfulqa"]:
                    print(f"Dataset truthfulqa, split {split}:")
                    print(f"  - Type: {type(datasets['truthfulqa'][split])}")
                    print(f"  - Column names: {datasets['truthfulqa'][split].column_names}")
                    print(f"  - Size: {len(datasets['truthfulqa'][split])}")
                    print(f"  - First example: {datasets['truthfulqa'][split][0]}")
                
            except Exception as e:
                print(f"Error loading TruthfulQA from CSV: {e}")
                print("Falling back to loading from HuggingFace...")
                
                # Load from HuggingFace
                truthfulqa_dataset = load_dataset("truthful_qa", "multiple_choice")
                
                # Process the HF dataset
                train_data = []
                eval_data = []
                
                # There's only a validation split in the HF dataset
                for i, sample in enumerate(truthfulqa_dataset["validation"]):
                    if i >= 2:  # Just load a few for training examples
                        break
                    
                    # Get the correct answer based on mc1_targets.labels
                    correct_idx = sample["mc1_targets"]["labels"].index(1)
                    correct_answer = sample["mc1_targets"]["choices"][correct_idx]
                    
                    train_data.append({
                        "question": sample["question"],
                        "answer": correct_answer,
                        "context": "",
                        "choices": sample["mc1_targets"]["choices"],
                        "labels": sample["mc1_targets"]["labels"],
                        "is_binary_choice": False
                    })
                
                # Process for evaluation (up to sample_limit)
                sample_limit = max_samples if max_samples else len(truthfulqa_dataset["validation"])
                
                for i, sample in enumerate(truthfulqa_dataset["validation"]):
                    if i >= sample_limit:
                        break
                    
                    # Get the correct answer based on mc1_targets.labels
                    correct_idx = sample["mc1_targets"]["labels"].index(1)
                    correct_answer = sample["mc1_targets"]["choices"][correct_idx]
                    
                    eval_data.append({
                        "question": sample["question"],
                        "answer": correct_answer,
                        "context": "",
                        "choices": sample["mc1_targets"]["choices"],
                        "labels": sample["mc1_targets"]["labels"],
                        "is_binary_choice": False
                    })
                
                # Create a dataset dictionary with train and eval splits
                datasets["truthfulqa"] = {
                    "train": Dataset.from_list(train_data),
                    "eval": Dataset.from_list(eval_data)
                }
            
        except Exception as e:
            print(f"Error loading TruthfulQA dataset: {e}")
            # Create an empty dataset
            datasets["truthfulqa"] = {
                "train": Dataset.from_list([]),
                "eval": Dataset.from_list([])
            }
    
    return datasets


def format_prompt(
    prompt_template: str, 
    example: Dict[str, Any]
) -> str:
    """
    Format a prompt using a template and example.
    
    Args:
        prompt_template: The template string with placeholders
        example: Dictionary with values to fill in the template
        
    Returns:
        Formatted prompt string
    """
    # Ensure all values in the example are strings
    for key, value in example.items():
        if not isinstance(value, str):
            example[key] = str(value)
    
    # Basic string formatting
    return prompt_template.format(**example)


def prepare_training_examples(
    examples: Dataset, 
    tokenizer: PreTrainedTokenizer, 
    prompt_template: str,
    input_key: str = "question",
    output_key: str = "support",
    context_key: str = None,
    max_input_length: int = None,
    max_output_length: int = None
) -> Dataset:
    """
    Prepare training examples by formatting prompts and tokenizing.
    
    Args:
        examples: Dataset of examples
        tokenizer: Tokenizer for the model
        prompt_template: Template for formatting prompts
        input_key: Key for input field in examples
        output_key: Key for output field in examples
        context_key: Optional key for context field in examples
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        
    Returns:
        Dataset object with processed examples
    """
    if max_input_length is None:
        max_input_length = DATA_PROCESSING["max_input_length"]
    if max_output_length is None:
        max_output_length = DATA_PROCESSING["max_output_length"]
    
    def process_example(example):
        # Prepare input dictionary for formatting
        input_dict = {
            "question": example[input_key]
        }
        
        # Add context if available
        if context_key and context_key in example:
            input_dict["context"] = example[context_key]
        
        # Format prompt
        prompt = format_prompt(prompt_template, input_dict)
        
        # Tokenize input
        tokenized_input = tokenizer(
            prompt,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    
        # Tokenize output
        tokenized_output = tokenizer(
            example[output_key],
            max_length=max_output_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    
        # Prepare final example
        result = {
            "input_ids": tokenized_input.input_ids[0],
            "attention_mask": tokenized_input.attention_mask[0],
            "labels": torch.where(
                tokenized_output.attention_mask[0] == 1,
                tokenized_output.input_ids[0],
                torch.tensor(-100, dtype=torch.long)
            )
        }
        
        return result
    
    # Process all examples
    processed_dataset = examples.map(
        process_example,
        remove_columns=examples.column_names
    )
    
    return processed_dataset


def calculate_hallucination_metrics(predictions, references):
    """
    Calculate metrics for hallucination detection.
    
    Args:
        predictions: List of model predictions
        references: List of ground truth references
        
    Returns:
        Dictionary of metrics
    """
    from rouge import Rouge
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    import nltk
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    rouge = Rouge()
    
    # Initialize metrics
    metrics = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bleu1": 0.0,
        "bleu4": 0.0,
        "empty_prediction_rate": 0.0,
        "hallucination_score": 0.0  # Higher means more hallucination
    }
    
    valid_pairs = 0
    
    for pred, ref in zip(predictions, references):
        # Skip empty predictions or references
        if not pred or not ref:
            metrics["empty_prediction_rate"] += 1
            continue
            
        valid_pairs += 1
        
        # Calculate ROUGE scores
        try:
            rouge_scores = rouge.get_scores(pred, ref)[0]
            metrics["rouge1"] += rouge_scores["rouge-1"]["f"]
            metrics["rouge2"] += rouge_scores["rouge-2"]["f"] 
            metrics["rougeL"] += rouge_scores["rouge-l"]["f"]
        except Exception:
            # Skip Rouge on error
            pass
        
        # Calculate BLEU scores
        try:
            reference_tokens = word_tokenize(ref.lower())
            prediction_tokens = word_tokenize(pred.lower())
            
            if reference_tokens and prediction_tokens:
                metrics["bleu1"] += sentence_bleu([reference_tokens], prediction_tokens, 
                                                weights=(1, 0, 0, 0))
                metrics["bleu4"] += sentence_bleu([reference_tokens], prediction_tokens, 
                                                weights=(0.25, 0.25, 0.25, 0.25))
        except Exception:
            # Skip BLEU on error
            pass
    
    # Average the metrics
    if valid_pairs > 0:
        metrics["rouge1"] /= valid_pairs
        metrics["rouge2"] /= valid_pairs
        metrics["rougeL"] /= valid_pairs
        metrics["bleu1"] /= valid_pairs
        metrics["bleu4"] /= valid_pairs
    
    # Calculate empty prediction rate
    metrics["empty_prediction_rate"] = metrics["empty_prediction_rate"] / len(predictions) if predictions else 0
    
    # Calculate hallucination score (1 - F1 score)
    # This is a simple approach - the lower the overlap with ground truth, the higher the hallucination
    metrics["hallucination_score"] = 1.0 - metrics["rougeL"]
    
    return metrics 