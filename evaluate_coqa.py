#!/usr/bin/env python3
"""
Simplified evaluation script focusing on just the CoQA dataset.
"""

import os
import sys
import logging
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
COQA_PROMPT_TEMPLATE = """[INST] Answer the following question based on the provided context:

Context: {context}

Question: {question} [/INST]"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test CoQA dataset loading and processing")
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of samples to process",
    )
    
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Name of the tokenizer to use",
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="coqa_evaluation_results.txt",
        help="File to save the results",
    )
    
    return parser.parse_args()


def load_coqa_dataset(max_samples=5):
    """
    Load and process CoQA dataset.
    
    Args:
        max_samples: Maximum number of samples to load
        
    Returns:
        Processed dataset
    """
    logger.info("Loading CoQA dataset...")
    
    try:
        # Load the dataset
        ds = load_dataset("stanfordnlp/coqa")
        
        # Process the dataset
        def process_coqa(example):
            """Process CoQA example to extract the first QA pair."""
            return {
                "question": example["questions"][0] if example["questions"] else "",
                "answer": example["answers"]["input_text"][0] if example["answers"]["input_text"] else "",
                "context": example["story"]
            }
        
        # Select a subset of the validation set
        eval_ds = ds["validation"].select(range(min(max_samples, len(ds["validation"]))))
        
        # Apply the processing function
        processed_ds = eval_ds.map(process_coqa, remove_columns=eval_ds.column_names)
        
        logger.info(f"Loaded and processed CoQA dataset with {len(processed_ds)} examples")
        return processed_ds
        
    except Exception as e:
        logger.error(f"Error loading CoQA dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def prepare_for_model(dataset, tokenizer, max_length=512):
    """
    Prepare dataset for model input.
    
    Args:
        dataset: Dataset to prepare
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Prepared dataset
    """
    logger.info("Preparing dataset for model input...")
    
    def format_example(example):
        """Format a single example."""
        # Format prompt
        prompt = COQA_PROMPT_TEMPLATE.format(
            context=example["context"],
            question=example["question"]
        )
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize output
        outputs = tokenizer(
            example["answer"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        result = {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": torch.where(
                outputs.attention_mask[0] == 1,
                outputs.input_ids[0],
                torch.tensor(-100, dtype=torch.long)
            )
        }
        
        # Keep the original text for reference
        result["question_text"] = example["question"]
        result["answer_text"] = example["answer"]
        result["context_text"] = example["context"]
        
        return result
    
    # Apply the formatting function
    prepared_ds = dataset.map(
        format_example,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Prepared {len(prepared_ds)} examples for model input")
    return prepared_ds


def main():
    """Main function."""
    args = parse_args()
    
    # Redirect output to file if specified
    if args.output_file:
        file_handler = logging.FileHandler(args.output_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return 1
    
    # Load and process CoQA dataset
    dataset = load_coqa_dataset(args.max_samples)
    if not dataset:
        logger.error("Failed to load dataset")
        return 1
    
    # Examine dataset structure
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Print a few examples
    for i, example in enumerate(dataset):
        logger.info(f"Example {i}:")
        logger.info(f"  Question: {example['question']}")
        logger.info(f"  Answer: {example['answer']}")
        logger.info(f"  Context (first 100 chars): {example['context'][:100]}...")
    
    # Prepare dataset for model
    prepared_ds = prepare_for_model(dataset, tokenizer)
    
    # Verify prepared dataset
    logger.info(f"Prepared dataset columns: {prepared_ds.column_names}")
    logger.info(f"First prepared example:")
    first_example = prepared_ds[0]
    logger.info(f"  Input shape: {first_example['input_ids'].shape}")
    logger.info(f"  Labels shape: {first_example['labels'].shape}")
    logger.info(f"  Original question: {first_example['question_text']}")
    logger.info(f"  Original answer: {first_example['answer_text']}")
    
    # Decode a tokenized input for verification
    input_text = tokenizer.decode(first_example["input_ids"], skip_special_tokens=True)
    logger.info(f"Decoded input (first 200 chars): {input_text[:200]}...")
    
    logger.info("Dataset processing completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 