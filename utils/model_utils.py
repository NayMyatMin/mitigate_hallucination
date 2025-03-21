"""
Utility functions for model loading, configuration and finetuning.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union

import torch
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

from config.model_config import MODELS, LORA_CONFIG, TRAINING_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

def load_base_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None
):
    """
    Load a base model and its tokenizer.
    
    Args:
        model_name: Name of the model in the MODELS config or a Hugging Face model ID
        device_map: Device mapping strategy
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        torch_dtype: Precision to load the model in (float16, float32, bfloat16)
        cache_dir: Directory to cache models
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if this is a direct HuggingFace model ID
    is_hf_model_id = any([
        model_name.startswith("meta-llama/"),  # Meta Llama models
        model_name.startswith("mistralai/"),   # Mistral models
        '/' in model_name                      # General pattern for org/model format
    ])
    
    if is_hf_model_id:
        logger.info(f"Using direct Hugging Face model ID: {model_name}")
        model_id = model_name
        tokenizer_id = model_name
    elif model_name in MODELS:
        model_config = MODELS[model_name]
        model_id = model_config["model_id"]
        tokenizer_id = model_config.get("tokenizer_id", model_id)
    else:
        available_models = list(MODELS.keys())
        logger.warning(f"Model {model_name} not found in config. Available configured models: {available_models}")
        logger.info(f"Attempting to load {model_name} directly from Hugging Face...")
        model_id = model_name
        tokenizer_id = model_name
    
    # Set up quantization config
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
            bnb_4bit_use_double_quant=True if load_in_4bit else False,
            bnb_4bit_quant_type="nf4" if load_in_4bit else None
        )
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            cache_dir=cache_dir,
            padding_side="right",
            use_fast=True,
        )
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        logger.info("Trying alternative tokenizer ID...")
        # Some models might use a different tokenizer than their model ID
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side="right",
            use_fast=True,
        )
    
    # Add special tokens and padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use provided dtype or default to bfloat16
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    
    # Load model
    logger.info(f"Loading model: {model_id} with {torch_dtype} precision")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    return model, tokenizer


def create_peft_config(
    lora_config: Optional[Dict[str, Any]] = None,
) -> LoraConfig:
    """
    Create a LoRA configuration for PEFT.
    
    Args:
        lora_config: Dictionary with LoRA configuration parameters
        
    Returns:
        LoraConfig object
    """
    if lora_config is None:
        lora_config = LORA_CONFIG
        
    return LoraConfig(**lora_config)


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    lora_config: Optional[Dict[str, Any]] = None,
) -> PeftModel:
    """
    Prepare a model for LoRA training.
    
    Args:
        model: Base model to prepare
        lora_config: LoRA configuration parameters
        
    Returns:
        Model prepared for training
    """
    # Create PEFT config
    peft_config = create_peft_config(lora_config)
    
    # Add LoRA adapter
    model = get_peft_model(model, peft_config)
    
    # Set up model for training
    model.enable_input_require_grads()
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def create_training_args(
    output_dir: str,
    training_config: Dict[str, Any],
    run_name: Optional[str] = None
) -> TrainingArguments:
    """
    Create training arguments from config.
    
    Args:
        output_dir: Output directory
        training_config: Training configuration parameters
        run_name: Optional run name
        
    Returns:
        Training arguments
    """
    config = training_config.copy()
    
    # Add required parameters
    config["output_dir"] = output_dir
    
    if run_name:
        config["run_name"] = run_name
    
    # Map parameters to their correct names for TrainingArguments
    parameter_mapping = {
        "max_seq_length": "max_length"  # Map max_seq_length to max_length
    }
    
    # Apply parameter mapping
    for old_param, new_param in parameter_mapping.items():
        if old_param in config:
            config[new_param] = config.pop(old_param)
    
    # Filter out parameters not supported by TrainingArguments
    # Get the valid parameters from TrainingArguments
    from inspect import signature
    valid_params = signature(TrainingArguments.__init__).parameters.keys()
    
    # Filter the config to only include valid parameters
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    
    # Create training arguments
    return TrainingArguments(**filtered_config)


def save_model_and_tokenizer(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    output_dir: str
):
    """
    Save a trained model and tokenizer.
    
    Args:
        model: Trained model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LoRA adapter weights
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to {output_dir}")


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    device_map: str = "auto",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None
):
    """
    Load a finetuned model with its adapter.
    
    Args:
        base_model_name: Name of the base model in MODELS config
        adapter_path: Path to the saved adapter
        device_map: Device mapping strategy
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        torch_dtype: Precision to load the model in (float16, float32, bfloat16)
        cache_dir: Directory to cache models
        
    Returns:
        Tuple of (model with adapter, tokenizer)
    """
    # Load base model and tokenizer
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=base_model_name,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir
    )
    
    # Use provided dtype or default to bfloat16
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    
    # Load adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=torch_dtype,
    )
    
    return model, tokenizer 