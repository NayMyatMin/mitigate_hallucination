#!/usr/bin/env python3
"""
Main training script for fine-tuning a model to mitigate hallucinations using LoRA.
"""

import os
import argparse
import logging
from datetime import datetime
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed
)
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import random

# Local imports
from config.model_config import MODELS, LORA_CONFIG, TRAINING_CONFIG, HALLUCINATION_CONFIG
from config.data_config import DATASETS, PROMPT_TEMPLATES, DATA_PROCESSING
from utils.data_utils import load_and_prepare_datasets, prepare_training_examples, mix_datasets
from utils.model_utils import (
    load_base_model_and_tokenizer, 
    prepare_model_for_training,
    create_training_args,
    save_model_and_tokenizer
)
from utils.evaluation_utils import evaluate_hallucination


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    """
    Custom trainer with modified loss function for hallucination mitigation.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation.
        
        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            return_outputs: If True, outputs will be returned along with the loss
            
        Returns:
            Loss or tuple (loss, outputs)
        """
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Standard language modeling loss (cross entropy)
        labels = inputs.get("labels")
        
        # Standard cross-entropy loss
        ce_loss = outputs.loss
        
        # Calculate additional hallucination penalty (example)
        # This is a placeholder - modify with your specific loss logic
        # For example, you could add:
        # 1. Confidence penalty for uncertain predictions
        # 2. Knowledge grounding loss 
        # 3. Citation relevance loss
        
        # Example: Add confidence penalty (softmax entropy regularization)
        # This penalizes overconfident predictions which can lead to hallucinations
        batch_size, seq_len, vocab_size = logits.shape
        
        # Only compute on non-padded tokens
        mask = (labels != -100).float()
        
        # Calculate token-wise entropy
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # We want to minimize entropy for tokens we're confident about (real knowledge)
        # but maximize it for tokens we're uncertain about (potential hallucinations)
        # This is just an example approach
        
        # Here we're using a simple regularization term - modify as needed
        entropy_reg = torch.mean(entropy * mask)
        entropy_weight = 0.1  # Adjust this weight as needed
        
        # Combined loss
        loss = ce_loss - entropy_weight * entropy_reg  # Negative because we want to maximize entropy
        
        # You can add more custom loss components here
        
        return (loss, outputs) if return_outputs else loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with LoRA for hallucination mitigation")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Llama-3.1-8B-Instruct",
        help="Name of the model to fine-tune (from config)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model checkpoints",
    )
    
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="mixed",
        choices=["factual", "citations", "hallucinations", "mixed"],
        help="Type of dataset to use for training",
    )
    
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Whether to load the model in 8-bit precision",
    )
    
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Whether to load the model in 4-bit precision",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides num_train_epochs)",
    )
    
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=0.1,
        help="Weight for entropy regularization in loss function",
    )
    
    return parser.parse_args()


def prepare_dataset_for_training(
    dataset_type: str,
    tokenizer: AutoTokenizer
) -> Dataset:
    """
    Prepare dataset for training based on dataset type.
    
    Args:
        dataset_type: Type of dataset to use (factual, citations, hallucinations, mixed)
        tokenizer: Tokenizer for the model
        
    Returns:
        Processed dataset ready for training
    """
    # Load datasets
    all_datasets = load_and_prepare_datasets(DATASETS)
    
    # Get the appropriate datasets based on type
    if dataset_type == "mixed":
        # Mix datasets according to the specified ratios
        train_datasets = {
            "factual": all_datasets["factual"]["train"],
            "citations": all_datasets["citations"]["train"],
            "hallucinations": all_datasets["hallucinations"]["train"]
        }
        
        eval_datasets = {
            "factual": all_datasets["factual"]["eval"],
            "citations": all_datasets["citations"]["eval"],
            "hallucinations": all_datasets["hallucinations"]["eval"]
        }
        
        # Mix training datasets
        mixed_train = mix_datasets(train_datasets, HALLUCINATION_CONFIG["data_mixing_ratio"])
        
        # Mix evaluation datasets
        mixed_eval = mix_datasets(eval_datasets, HALLUCINATION_CONFIG["data_mixing_ratio"])
        
        dataset = {
            "train": mixed_train,
            "eval": mixed_eval
        }
    else:
        # Use a single dataset type
        dataset = {
            "train": all_datasets[dataset_type]["train"],
            "eval": all_datasets[dataset_type]["eval"]
        }
    
    # Get the appropriate template
    prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["factual"])
    
    # Prepare examples for training
    input_key = DATASETS[dataset_type]["input_key"]
    output_key = DATASETS[dataset_type]["output_key"]
    
    train_dataset = prepare_training_examples(
        dataset["train"],
        tokenizer,
        prompt_template,
        input_key=input_key,
        output_key=output_key
    )
    
    eval_dataset = prepare_training_examples(
        dataset["eval"],
        tokenizer,
        prompt_template,
        input_key=input_key,
        output_key=output_key
    )
    
    return {"train": train_dataset, "eval": eval_dataset}


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Entropy weight: {args.entropy_weight}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=args.model_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # Prepare model for LoRA training
    logger.info("Preparing model for LoRA training...")
    model = prepare_model_for_training(model)
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    try:
        dataset = prepare_dataset_for_training(args.dataset_type, tokenizer)
        
        # Verify that the datasets are not empty
        if not dataset["train"] or not dataset["eval"] or len(dataset["train"]["input_ids"]) == 0:
            logger.error("Training dataset is empty. Check dataset configuration.")
            logger.info("Creating a small synthetic dataset for testing...")
            
            # Create a small synthetic dataset for testing the training pipeline
            synthetic_data = create_synthetic_dataset(tokenizer, 100)
            dataset["train"] = synthetic_data
            dataset["eval"] = synthetic_data
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        logger.info("Creating a small synthetic dataset for testing...")
        
        # Create a small synthetic dataset for testing the training pipeline
        synthetic_data = create_synthetic_dataset(tokenizer, 100)
        dataset = {
            "train": synthetic_data,
            "eval": synthetic_data
        }
    
    # Create training arguments
    training_config = TRAINING_CONFIG.copy()
    
    if args.max_steps is not None:
        training_config["max_steps"] = args.max_steps
        # Use steps-based training
        training_config["num_train_epochs"] = 100  # Set to a large number
    
    # Create run name
    run_name = f"{args.model_name}_{args.dataset_type}_lora_r{LORA_CONFIG['r']}"
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=output_dir,
        training_config=training_config,
        run_name=run_name
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest"
    )
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    logger.info("Saving model and tokenizer...")
    save_model_and_tokenizer(model, tokenizer, output_dir)
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Log the evaluation results
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Training completed. Model saved to {output_dir}")


def create_synthetic_dataset(tokenizer, num_samples=100):
    """
    Create a small synthetic dataset for testing the training pipeline.
    
    Args:
        tokenizer: The tokenizer to use
        num_samples: Number of synthetic samples to create
        
    Returns:
        Dictionary with tokenized synthetic data
    """
    # Generate simple question-answer pairs
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "What is the tallest mountain in the world?",
        "Who was the first person to walk on the moon?"
    ]
    
    answers = [
        "The capital of France is Paris.",
        "William Shakespeare wrote Romeo and Juliet.",
        "The speed of light is approximately 299,792,458 meters per second.",
        "Mount Everest is the tallest mountain in the world.",
        "Neil Armstrong was the first person to walk on the moon."
    ]
    
    # Create input prompts with template
    prompts = []
    targets = []
    
    for _ in range(num_samples):
        idx = random.randint(0, len(questions) - 1)
        
        prompt_template = PROMPT_TEMPLATES["factual"]
        prompt = prompt_template.format(question=questions[idx])
        target = answers[idx]
        
        prompts.append(prompt)
        targets.append(target)
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(
        prompts,
        max_length=DATA_PROCESSING["max_input_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Tokenize targets
    tokenized_targets = tokenizer(
        targets,
        max_length=DATA_PROCESSING["max_output_length"],
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


if __name__ == "__main__":
    main() 