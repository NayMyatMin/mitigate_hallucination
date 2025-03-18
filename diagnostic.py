#!/usr/bin/env python3
"""
Diagnostic script to identify specific errors in the training pipeline.
"""

import os
import sys
import traceback
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports."""
    logger.info("Testing imports...")
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info("Transformers imported successfully")
        
        from peft import LoraConfig, get_peft_model
        logger.info("PEFT imported successfully")
        
        from datasets import load_dataset, Dataset
        logger.info("Datasets imported successfully")
        
        # Local imports
        from config.model_config import MODELS, LORA_CONFIG, TRAINING_CONFIG
        logger.info("Model config imported successfully")
        
        from config.data_config import DATASETS, PROMPT_TEMPLATES, DATA_PROCESSING
        logger.info("Data config imported successfully")
        
        from utils.data_utils import load_and_prepare_datasets, prepare_training_examples
        logger.info("Data utils imported successfully")
        
        from utils.model_utils import load_base_model_and_tokenizer, prepare_model_for_training
        logger.info("Model utils imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"Import error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_dataset_loading():
    """Test dataset loading."""
    logger.info("Testing dataset loading...")
    try:
        from datasets import load_dataset
        
        # Try loading SciQ dataset
        logger.info("Loading SciQ dataset...")
        ds = load_dataset("sciq")
        logger.info(f"SciQ dataset loaded successfully: {ds}")
        logger.info(f"Available splits: {ds.keys()}")
        
        if "train" in ds:
            logger.info(f"Train examples: {len(ds['train'])}")
            logger.info(f"First example: {ds['train'][0]}")
        
        if "validation" in ds:
            logger.info(f"Validation examples: {len(ds['validation'])}")
        
        return True
    except Exception as e:
        logger.error(f"Dataset loading error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_data_processing():
    """Test data processing utilities."""
    logger.info("Testing data processing...")
    try:
        from utils.data_utils import load_and_prepare_datasets
        from config.data_config import DATASETS, PROMPT_TEMPLATES
        
        # Load datasets
        logger.info("Loading and preparing datasets...")
        all_datasets = load_and_prepare_datasets(DATASETS)
        logger.info(f"Datasets loaded: {list(all_datasets.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"Data processing error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_model_loading(verbose=False):
    """Test model loading (without full download)."""
    logger.info("Testing model access (validating API connection)...")
    try:
        from config.model_config import MODELS
        model_name = "llama3.1-8b-instruct"
        model_config = MODELS[model_name]
        
        if verbose:
            # This will actually attempt to download the model - may take a while
            logger.info(f"Attempting to load tokenizer for {model_config['tokenizer_id']}...")
            tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_id"])
            logger.info("Tokenizer loaded successfully")
        else:
            # Just validate the model ID is accessible
            logger.info(f"Checking model access for {model_config['model_id']}...")
            import huggingface_hub
            model_info = huggingface_hub.model_info(model_config["model_id"])
            logger.info(f"Model {model_config['model_id']} is accessible: {model_info.id}")
        
        return True
    except Exception as e:
        logger.error(f"Model access error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_tokenization():
    """Test tokenization process."""
    logger.info("Testing tokenization...")
    try:
        from config.data_config import PROMPT_TEMPLATES, DATA_PROCESSING
        
        # Load a small tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Create a test prompt
        prompt_template = PROMPT_TEMPLATES["citations"]
        prompt = prompt_template.format(question="What is the capital of France?")
        
        logger.info(f"Test prompt: {prompt}")
        
        # Tokenize
        tokenized_inputs = tokenizer(
            prompt,
            max_length=DATA_PROCESSING["max_input_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        logger.info(f"Tokenization successful, shape: {tokenized_inputs.input_ids.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        logger.error(traceback.format_exc())
        return False

def run_diagnostics():
    """Run all diagnostic tests."""
    logger.info("Running diagnostics...")
    
    results = {
        "imports": test_imports(),
        "dataset_loading": test_dataset_loading(),
        "data_processing": test_data_processing(),
        "model_access": test_model_loading(verbose=False),
        "tokenization": test_tokenization()
    }
    
    logger.info("Diagnostics complete.")
    logger.info("Results:")
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test}: {status}")
    
    # Print a summary of what might be wrong
    failed_tests = [test for test, passed in results.items() if not passed]
    if failed_tests:
        logger.info("\nPossible issues:")
        if "imports" in failed_tests:
            logger.info("- Missing dependencies: Try running 'pip install -r requirements.txt'")
        if "dataset_loading" in failed_tests:
            logger.info("- SciQ dataset access issues: Check internet connection or try a different dataset")
        if "data_processing" in failed_tests:
            logger.info("- Data processing utilities may have errors: Check utils/data_utils.py")
        if "model_access" in failed_tests:
            logger.info("- Model access issue: Check HuggingFace credentials and model permissions")
            logger.info("  You might need to run 'huggingface-cli login' or use a different model")
        if "tokenization" in failed_tests:
            logger.info("- Tokenization error: Check your prompt templates in config/data_config.py")
    else:
        logger.info("\nAll components seem to be working correctly in isolation.")
        logger.info("The issue might be in the interaction between components.")

if __name__ == "__main__":
    run_diagnostics() 