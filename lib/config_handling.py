"""
Configuration and setup utilities for hallucination evaluation.
"""

import os
import json
import logging
import torch
import argparse
from transformers import set_seed
from peft import PeftModel
from typing import Dict, Any, Tuple, Optional

from utils.model_utils import load_base_model_and_tokenizer
from utils.eval_utils import parse_args, configure_model_for_generation

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to use verbose (DEBUG) logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setup_output_directory(model_name: str, adapter_path: Optional[str] = None, 
                          base_dir: str = "evaluation_results") -> str:
    """
    Set up the output directory for evaluation results.
    
    Args:
        model_name: Name of the model
        adapter_path: Path to LoRA adapter (if applicable)
        base_dir: Base directory for results
        
    Returns:
        Path to the output directory
    """
    model_name_short = model_name.replace("/", "_")
    adapter_suffix = "_lora" if adapter_path else ""
    output_dir = os.path.join(base_dir, f"{model_name_short}{adapter_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(config: Dict[str, Any], output_dir: str):
    """
    Save evaluation configuration to a file.
    
    Args:
        config: Dictionary of configuration parameters
        output_dir: Directory to save configuration
    """
    config_file = os.path.join(output_dir, "evaluation_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def get_torch_dtype(dtype_str: str) -> Optional[torch.dtype]:
    """
    Convert string dtype to torch dtype.
    
    Args:
        dtype_str: String representation of dtype
        
    Returns:
        torch.dtype or None
    """
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    return None


def setup_model(model_name: str, adapter_path: Optional[str] = None, 
               load_in_8bit: bool = False, load_in_4bit: bool = False,
               torch_dtype: Optional[str] = "float16", max_new_tokens: int = 256):
    """
    Set up model and tokenizer for evaluation.
    
    Args:
        model_name: Name of the model to load
        adapter_path: Path to LoRA adapter
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        torch_dtype: Precision for model loading
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Convert string dtype to torch dtype
    dtype = get_torch_dtype(torch_dtype)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        torch_dtype=dtype
    )
    
    # Load LoRA adapter if specified
    if adapter_path:
        logger.info(f"Loading LoRA adapter from {adapter_path}...")
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("LoRA adapter loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading LoRA adapter: {e}")
            logger.info("Proceeding with base model only.")
    
    # Configure model for generation
    model.eval()  # Set model to evaluation mode
    model = configure_model_for_generation(model, max_new_tokens)
    
    return model, tokenizer


def log_evaluation_start(args: argparse.Namespace):
    """
    Log evaluation start parameters.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA adapter: {args.adapter_path if args.adapter_path else 'None'}")
    logger.info(f"Datasets: {args.dataset_types}")
    logger.info(f"Max eval samples per dataset: {args.max_eval_samples}")
    logger.info(f"Clean predictions: {args.clean_predictions}")
    if hasattr(args, 'use_gpt4o_mini') and args.use_gpt4o_mini:
        logger.info("Using GPT-4o-mini as a judge for hallucination evaluation") 