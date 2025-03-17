"""
Utility functions for data processing and preparation.
"""

import os
import json
from typing import Dict, List, Union, Any, Tuple
from datasets import load_dataset, Dataset, DatasetDict
import torch
from transformers import PreTrainedTokenizer

from config.data_config import DATASETS, PROMPT_TEMPLATES, DATA_PROCESSING


def load_and_prepare_datasets(dataset_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dataset]]:
    """
    Load and prepare datasets based on configurations.
    
    Args:
        dataset_configs: Dictionary of dataset configurations
        
    Returns:
        Dictionary of prepared datasets
    """
    datasets = {}
    
    for dataset_name, config in dataset_configs.items():
        try:
            # Handle specific datasets based on their structure
            if dataset_name == "factual":
                # TruthfulQA dataset
                subset = config.get("subset", "mc")
                try:
                    ds = load_dataset(config["train"], subset)
                    
                    # Process TruthfulQA depending on the dataset structure
                    if isinstance(ds, DatasetDict):
                        if "validation" in ds:
                            train_ds = ds["validation"].select(range(min(300, len(ds["validation"]))))
                            eval_ds = ds["validation"].select(range(300, min(400, len(ds["validation"]))))
                        else:
                            ds = ds["train"].train_test_split(
                                test_size=0.2, 
                                seed=DATA_PROCESSING["shuffle_seed"]
                            )
                            train_ds = ds["train"]
                            eval_ds = ds["test"]
                        
                        # Handle different TruthfulQA structures
                        def process_truthfulqa(example):
                            # Check the structure and extract what we need
                            if "mc1_targets" in example:
                                # For MC1 format
                                return {
                                    "question": example["question"],
                                    "best_answer": example["mc1_targets"]["labels"][0] if example["mc1_targets"]["labels"] else "No correct answer available."
                                }
                            elif "mc2_targets" in example:
                                # For MC2 format
                                return {
                                    "question": example["question"],
                                    "best_answer": example["mc2_targets"]["choices"][0] if example["mc2_targets"]["choices"] else "No correct answer available."
                                }
                            elif "correct_answers" in example:
                                # Original format
                                return {
                                    "question": example["question"],
                                    "best_answer": example["correct_answers"][0] if example["correct_answers"] else "No correct answer available."
                                }
                            else:
                                # Fallback if we can't find the expected structure
                                return {
                                    "question": example.get("question", ""),
                                    "best_answer": "No answer available in dataset."
                                }
                        
                        train_ds = train_ds.map(process_truthfulqa)
                        eval_ds = eval_ds.map(process_truthfulqa)
                except Exception as e:
                    # Fallback to a simple QA dataset if TruthfulQA fails
                    print(f"Failed to load TruthfulQA with error: {e}")
                    print("Falling back to SQUAD dataset")
                    ds = load_dataset("squad")
                    
                    train_ds = ds["train"].select(range(min(300, len(ds["train"]))))
                    eval_ds = ds["validation"].select(range(min(100, len(ds["validation"]))))
                    
                    def process_squad(example):
                        return {
                            "question": example["question"],
                            "best_answer": example["answers"]["text"][0] if example["answers"]["text"] else "No answer available."
                        }
                    
                    train_ds = train_ds.map(process_squad)
                    eval_ds = eval_ds.map(process_squad)
                    
            elif dataset_name == "hallucinations":
                try:
                    # Try to load the configured hallucination dataset
                    ds = load_dataset(config["train"])
                    
                    if isinstance(ds, DatasetDict):
                        if "train" in ds:
                            train_ds = ds["train"].select(range(min(300, len(ds["train"]))))
                            if "validation" in ds:
                                eval_ds = ds["validation"].select(range(min(100, len(ds["validation"]))))
                            else:
                                # Take a portion of training data for evaluation
                                split = ds["train"].train_test_split(
                                    test_size=0.2, 
                                    seed=DATA_PROCESSING["shuffle_seed"]
                                )
                                train_ds = split["train"]
                                eval_ds = split["test"]
                        else:
                            # Use a default split if no train/test is available
                            split = ds["test" if "test" in ds else list(ds.keys())[0]].train_test_split(
                                test_size=0.2, 
                                seed=DATA_PROCESSING["shuffle_seed"]
                            )
                            train_ds = split["train"].select(range(min(300, len(split["train"]))))
                            eval_ds = split["test"].select(range(min(100, len(split["test"]))))
                    else:
                        # Single split dataset
                        split = ds.train_test_split(
                            test_size=0.2, 
                            seed=DATA_PROCESSING["shuffle_seed"]
                        )
                        train_ds = split["train"].select(range(min(300, len(split["train"]))))
                        eval_ds = split["test"].select(range(min(100, len(split["test"]))))
                    
                    # Process for the OpenAI summarize_from_feedback dataset
                    def process_hallucination(example):
                        if dataset_name == "openai/summarize_from_feedback":
                            # Structure for the OpenAI dataset
                            if "prompt" in example and "summaries" in example and len(example["summaries"]) > 0:
                                return {
                                    "query": example["prompt"],
                                    "response": example["summaries"][0]["text"] if example["summaries"] else ""
                                }
                            else:
                                return {
                                    "query": example.get("prompt", example.get("query", example.get("question", ""))),
                                    "response": example.get("response", "")
                                }
                        else:
                            # Generic handling for other datasets
                            return {
                                "query": example.get("query", example.get("question", example.get("prompt", ""))),
                                "response": example.get("response", example.get("answer", example.get("summary", "")))
                            }
                    
                    train_ds = train_ds.map(process_hallucination)
                    eval_ds = eval_ds.map(process_hallucination)
                    
                except Exception as e:
                    print(f"Failed to load hallucination dataset: {e}")
                    print("Falling back to eli5 dataset")
                    
                    # Fallback to ELI5 dataset which has question-answer pairs
                    ds = load_dataset("eli5", "askscience_ama")
                    
                    train_ds = ds["train_asks"].select(range(min(300, len(ds["train_asks"]))))
                    eval_ds = ds["validation_asks"].select(range(min(100, len(ds["validation_asks"]))))
                    
                    def process_eli5(example):
                        return {
                            "query": example.get("title", ""),
                            "response": example.get("answers", {}).get("text", [""])[0] if example.get("answers", {}).get("text", []) else ""
                        }
                    
                    train_ds = train_ds.map(process_eli5)
                    eval_ds = eval_ds.map(process_eli5)
                    
            elif dataset_name == "citations":
                # SciQ dataset
                try:
                    ds = load_dataset(config["train"])
                    
                    if isinstance(ds, DatasetDict):
                        if "train" in ds:
                            train_ds = ds["train"].select(range(min(300, len(ds["train"]))))
                            if "validation" in ds:
                                eval_ds = ds["validation"].select(range(min(100, len(ds["validation"]))))
                            else:
                                eval_ds = ds["test" if "test" in ds else "train"].select(range(min(100, len(ds["test" if "test" in ds else "train"]))))
                        else:
                            # Use a default split if no train/test is available
                            ds_split = list(ds.keys())[0]
                            split = ds[ds_split].train_test_split(
                                test_size=0.2, 
                                seed=DATA_PROCESSING["shuffle_seed"]
                            )
                            train_ds = split["train"].select(range(min(300, len(split["train"]))))
                            eval_ds = split["test"].select(range(min(100, len(split["test"]))))
                    else:
                        # Single split dataset
                        split = ds.train_test_split(
                            test_size=0.2, 
                            seed=DATA_PROCESSING["shuffle_seed"]
                        )
                        train_ds = split["train"].select(range(min(300, len(split["train"]))))
                        eval_ds = split["test"].select(range(min(100, len(split["test"]))))
                    
                    # Process SciQ to ensure all required fields
                    def process_sciq(example):
                        return {
                            "question": example.get("question", ""),
                            "support": example.get("support", "")  # Scientific support/citation
                        }
                    
                    train_ds = train_ds.map(process_sciq)
                    eval_ds = eval_ds.map(process_sciq)
                except Exception as e:
                    print(f"Failed to load citations dataset: {e}")
                    # If sciq fails, try to use another dataset that has citations/references
                    print("No fallback dataset available for citations, using empty dataset")
                    
                    # Create an empty dataset with the required structure
                    train_ds = Dataset.from_dict({"question": [], "support": []})
                    eval_ds = Dataset.from_dict({"question": [], "support": []})
                
            else:
                # Generic dataset loading
                if os.path.exists(config["train"]):
                    # Load from local file
                    train_ds = load_dataset(
                        config["format"], 
                        data_files=config["train"], 
                        split="train"
                    )
                    eval_ds = load_dataset(
                        config["format"], 
                        data_files=config["eval"], 
                        split="train"
                    )
                else:
                    # Load from Hugging Face Hub
                    ds = load_dataset(config["train"])
                    if isinstance(ds, DatasetDict) and "train" in ds and "validation" in ds:
                        train_ds = ds["train"]
                        eval_ds = ds["validation"]
                    else:
                        # Split dataset if it doesn't have predefined splits
                        ds = ds.train_test_split(
                            test_size=1.0 - DATA_PROCESSING["train_test_split"],
                            seed=DATA_PROCESSING["shuffle_seed"]
                        )
                        train_ds = ds["train"]
                        eval_ds = ds["test"]
            
            datasets[dataset_name] = {
                "train": train_ds,
                "eval": eval_ds
            }
            
            print(f"Loaded {dataset_name} dataset:")
            print(f"  - Train size: {len(train_ds)}")
            print(f"  - Eval size: {len(eval_ds)}")
            
        except Exception as e:
            print(f"Error loading {dataset_name} dataset: {e}")
            # Provide an empty dataset as fallback
            datasets[dataset_name] = {
                "train": Dataset.from_dict({"question": [], config["output_key"]: []}),
                "eval": Dataset.from_dict({"question": [], config["output_key"]: []})
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
    # Basic string formatting
    return prompt_template.format(**example)


def prepare_training_examples(
    examples: Dict[str, List[Any]], 
    tokenizer: PreTrainedTokenizer, 
    prompt_template: str,
    input_key: str = "input",
    output_key: str = "output",
    max_input_length: int = None,
    max_output_length: int = None
) -> Dict[str, List[Any]]:
    """
    Prepare training examples by formatting prompts and tokenizing.
    
    Args:
        examples: Dictionary of examples
        tokenizer: Tokenizer for the model
        prompt_template: Template for formatting prompts
        input_key: Key for input field in examples
        output_key: Key for output field in examples
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        
    Returns:
        Dictionary with processed examples
    """
    if max_input_length is None:
        max_input_length = DATA_PROCESSING["max_input_length"]
    if max_output_length is None:
        max_output_length = DATA_PROCESSING["max_output_length"]
    
    inputs = []
    targets = []
    
    # Ensure the input and output keys exist in the dataset
    if input_key not in examples or output_key not in examples:
        available_keys = list(examples.keys())
        print(f"Warning: Required keys {input_key} or {output_key} not in dataset. Available keys: {available_keys}")
        
        # Try to use alternative keys if available
        if input_key not in examples and "question" in examples:
            input_key = "question"
        elif input_key not in examples and "query" in examples:
            input_key = "query"
            
        if output_key not in examples and "answer" in examples:
            output_key = "answer"
        elif output_key not in examples and "best_answer" in examples:
            output_key = "best_answer"
        elif output_key not in examples and "support" in examples:
            output_key = "support"
        elif output_key not in examples and "response" in examples:
            output_key = "response"
    
    for i in range(len(examples[input_key])):
        example_dict = {
            "question": examples[input_key][i]
        }
        
        # For hallucination examples, add bad example if present
        if "bad_example" in examples:
            example_dict["bad_example"] = examples["bad_example"][i]
            
        prompt = format_prompt(prompt_template, example_dict)
        target = examples[output_key][i]
        
        inputs.append(prompt)
        targets.append(target)
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Tokenize targets
    tokenized_targets = tokenizer(
        targets,
        max_length=max_output_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    result = {
        "input_ids": tokenized_inputs.input_ids,
        "attention_mask": tokenized_inputs.attention_mask,
        "labels": tokenized_targets.input_ids,
    }
    
    # Replace padding token id with -100 in labels for loss calculation
    result["labels"] = torch.where(
        tokenized_targets.attention_mask == 1,
        result["labels"],
        -100
    )
    
    return result


def mix_datasets(
    datasets: Dict[str, Dataset], 
    mixing_ratios: Dict[str, float]
) -> Dataset:
    """
    Mix multiple datasets according to specified ratios.
    
    Args:
        datasets: Dictionary of datasets to mix
        mixing_ratios: Dictionary specifying the ratio for each dataset
        
    Returns:
        Mixed dataset
    """
    # First, filter out empty datasets
    non_empty_datasets = {k: v for k, v in datasets.items() if len(v) > 0}
    
    # If all datasets are empty, return an empty dataset
    if not non_empty_datasets:
        print("Warning: All datasets are empty. Creating an empty mixed dataset.")
        return Dataset.from_dict({"question": [], "answer": []})
    
    # Map dataset names to mixing ratio keys (e.g., "factual" -> "factual_data")
    ratio_key_mapping = {
        "factual": "factual_data",
        "citations": "citation_data",
        "hallucinations": "hallucination_examples"
    }
    
    # Filter mixing ratios to only include available datasets
    available_ratios = {}
    for ds_name in non_empty_datasets:
        # Get the corresponding ratio key or use the dataset name as a fallback
        ratio_key = ratio_key_mapping.get(ds_name, ds_name)
        if ratio_key in mixing_ratios:
            available_ratios[ds_name] = mixing_ratios[ratio_key]
        else:
            # If no matching ratio is found, assign a default ratio
            available_ratios[ds_name] = 1.0
    
    # If no valid ratios, assign equal ratios
    if not available_ratios:
        available_ratios = {k: 1.0 for k in non_empty_datasets}
    
    # Normalize ratios
    total = sum(available_ratios.values())
    normalized_ratios = {k: v / total for k, v in available_ratios.items()}
    
    # Calculate samples from each dataset
    min_dataset_size = min([len(ds) for ds in non_empty_datasets.values()])
    total_samples = min_dataset_size
    
    # Calculate samples per dataset
    dataset_samples = {
        k: max(1, int(normalized_ratios[k] * total_samples))
        for k in normalized_ratios
    }
    
    # Adjust for rounding errors
    remaining = total_samples - sum(dataset_samples.values())
    if remaining > 0 and dataset_samples:
        # Add remaining samples to the dataset with the highest ratio
        max_ratio_key = max(dataset_samples.keys(), key=lambda k: normalized_ratios[k])
        dataset_samples[max_ratio_key] += remaining
    
    # Sample and concatenate datasets
    mixed_datasets = []
    for ds_name, num_samples in dataset_samples.items():
        if num_samples > 0 and ds_name in non_empty_datasets:
            # Shuffle and select samples
            shuffled_ds = non_empty_datasets[ds_name].shuffle(seed=DATA_PROCESSING["shuffle_seed"])
            # Ensure we don't select more samples than available
            actual_samples = min(num_samples, len(shuffled_ds))
            if actual_samples > 0:
                mixed_datasets.append(shuffled_ds.select(range(actual_samples)))
    
    # Concatenate all datasets
    if mixed_datasets:
        # Get the common features across all datasets
        common_features = set.intersection(*[set(ds.features.keys()) for ds in mixed_datasets])
        
        # Create a new dictionary with only the common features
        mixed_data = {}
        for feature in common_features:
            mixed_data[feature] = []
            for ds in mixed_datasets:
                mixed_data[feature].extend(ds[feature])
        
        return Dataset.from_dict(mixed_data)
    else:
        # Create an empty dataset with a reasonable structure
        return Dataset.from_dict({"question": [], "answer": []}) 