#!/usr/bin/env python3
"""
Comprehensive dataset checking script to verify the loading and processing 
of all hallucination evaluation datasets.
"""

import os
import sys
import logging
import time
import argparse
from datasets import load_dataset, Dataset, DownloadConfig
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_check_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_coqa():
    """Check CoQA dataset."""
    logger.info("======== Checking CoQA Dataset ========")
    try:
        # Load the dataset directly
        logger.info("Loading CoQA dataset...")
        ds = load_dataset("stanfordnlp/coqa")
        
        # Check structure
        logger.info(f"Dataset splits: {list(ds.keys())}")
        for split in ds.keys():
            logger.info(f"Split '{split}' size: {len(ds[split])}")
            logger.info(f"Features: {ds[split].column_names}")
        
        # Check a sample
        if "validation" in ds and len(ds["validation"]) > 0:
            sample = ds["validation"][0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            logger.info(f"Sample questions (type): {type(sample.get('questions', ''))} (length: {len(sample.get('questions', []))})")
            logger.info(f"Sample first question: {sample.get('questions', [''])[0] if sample.get('questions', []) else ''}")
            logger.info(f"Sample answers structure: {list(sample.get('answers', {}).keys()) if isinstance(sample.get('answers', {}), dict) else type(sample.get('answers', {}))}") 
        
        # Process with our function
        logger.info("Processing with our functions...")
        from utils.data_utils import load_and_prepare_datasets
        from config.data_config import DATASETS
        
        datasets = load_and_prepare_datasets({"coqa": DATASETS["coqa"]}, max_samples=2)
        
        if "coqa" in datasets:
            for split in datasets["coqa"]:
                logger.info(f"Processed '{split}' size: {len(datasets['coqa'][split])}")
                logger.info(f"Processed columns: {datasets['coqa'][split].column_names}")
        
        logger.info("CoQA dataset check: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"Error checking CoQA dataset: {e}")
        logger.error(traceback.format_exc())
        logger.info("CoQA dataset check: FAILED")
        return False

def check_squad_v2():
    """Check SQuAD v2.0 dataset."""
    logger.info("======== Checking SQuAD v2.0 Dataset ========")
    try:
        # Load the dataset directly
        logger.info("Loading SQuAD v2.0 dataset...")
        ds = load_dataset("squad_v2")
        
        # Check structure
        logger.info(f"Dataset splits: {list(ds.keys())}")
        for split in ds.keys():
            logger.info(f"Split '{split}' size: {len(ds[split])}")
            logger.info(f"Features: {ds[split].column_names}")
        
        # Check a sample
        if "validation" in ds and len(ds["validation"]) > 0:
            sample = ds["validation"][0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            logger.info(f"Sample question: {sample.get('question', '')}")
            logger.info(f"Sample is_impossible: {sample.get('is_impossible', '')}")
            logger.info(f"Sample answers structure: {type(sample.get('answers', {}))} with keys: {list(sample.get('answers', {}).keys()) if isinstance(sample.get('answers', {}), dict) else 'N/A'}")
        
        # Process with our function
        logger.info("Processing with our functions...")
        from utils.data_utils import load_and_prepare_datasets
        from config.data_config import DATASETS
        
        datasets = load_and_prepare_datasets({"squad_v2": DATASETS["squad_v2"]}, max_samples=2)
        
        if "squad_v2" in datasets:
            for split in datasets["squad_v2"]:
                logger.info(f"Processed '{split}' size: {len(datasets['squad_v2'][split])}")
                logger.info(f"Processed columns: {datasets['squad_v2'][split].column_names}")
                if len(datasets["squad_v2"][split]) > 0:
                    sample = datasets["squad_v2"][split][0]
                    logger.info(f"Processed sample: {sample}")
        
        logger.info("SQuAD v2.0 dataset check: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"Error checking SQuAD v2.0 dataset: {e}")
        logger.error(traceback.format_exc())
        logger.info("SQuAD v2.0 dataset check: FAILED")
        return False

def check_triviaqa():
    """Check TriviaQA dataset."""
    logger.info("======== Checking TriviaQA Dataset ========")
    try:
        # Load the dataset directly
        logger.info("Loading TriviaQA dataset...")
        ds = load_dataset("trivia_qa", "rc.nocontext")
        
        # Check structure
        logger.info(f"Dataset splits: {list(ds.keys())}")
        for split in ds.keys():
            logger.info(f"Split '{split}' size: {len(ds[split])}")
            logger.info(f"Features: {ds[split].column_names}")
        
        # Check a sample
        if "validation" in ds and len(ds["validation"]) > 0:
            sample = ds["validation"][0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            logger.info(f"Sample question: {sample.get('question', '')}")
            logger.info(f"Sample answer: {type(sample.get('answer', {}))} with keys: {list(sample.get('answer', {}).keys()) if isinstance(sample.get('answer', {}), dict) else 'N/A'}")
        
        # Process with our function
        logger.info("Processing with our functions...")
        from utils.data_utils import load_and_prepare_datasets
        from config.data_config import DATASETS
        
        datasets = load_and_prepare_datasets({"triviaqa": DATASETS["triviaqa"]}, max_samples=2)
        
        if "triviaqa" in datasets:
            for split in datasets["triviaqa"]:
                logger.info(f"Processed '{split}' size: {len(datasets['triviaqa'][split])}")
                logger.info(f"Processed columns: {datasets['triviaqa'][split].column_names}")
        
        logger.info("TriviaQA dataset check: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"Error checking TriviaQA dataset: {e}")
        logger.error(traceback.format_exc())
        logger.info("TriviaQA dataset check: FAILED")
        return False

# def check_nq():
#     """Check Natural Questions dataset."""
#     logger.info("======== Checking Natural Questions Dataset ========")
#     try:
#         # Load original dataset to check structure
#         logger.info("Loading Natural Questions dataset...")
#         from datasets import load_dataset
#         
#         try:
#             # Try loading from HF
#             nq_dataset = load_dataset("natural_questions", streaming=True)
#             
#             # Get some basic info
#             batch = next(iter(nq_dataset["train"].take(1)))
#             logger.info(f"Sample keys: {list(batch.keys())}")
#             
#             # Check for required fields
#             if "question" in batch and "annotations" in batch:
#                 logger.info(f"Sample question: {batch['question']['text']}")
#                 
#                 annotation = batch["annotations"][0] if batch["annotations"] else None
#                 if annotation and "short_answers" in annotation:
#                     logger.info(f"Sample short_answers: {annotation['short_answers']}")
#                 
#                 if annotation and "yes_no_answer" in annotation:
#                     logger.info(f"Sample yes_no_answer: {annotation['yes_no_answer']}")
#             
#         except Exception as e:
#             logger.info(f"Error loading from HF: {e}")
#             logger.info("Creating synthetic sample for structure check")
#             
#             # Create a synthetic sample
#             class BatchSynth:
#                 def __init__(self):
#                     self.data = {
#                         "question": {"text": "What is the capital of France?"},
#                         "document": {"html": "<html><body>Paris is the capital of France.</body></html>"},
#                         "annotations": [
#                             {"short_answers": ["Paris"], "yes_no_answer": "NONE"}
#                         ]
#                     }
#                 
#                 def __getitem__(self, key):
#                     return self.data.get(key)
#                 
#                 def keys(self):
#                     return self.data.keys()
#             
#             batch = BatchSynth()
#             logger.info(f"Synthetic sample keys: {list(batch.keys())}")
#         
#         # Now test our processing functions by loading
#         datasets = load_and_prepare_datasets({"nq": DATASETS["nq"]}, max_samples=2)
#         
#         if "nq" in datasets:
#             for split in datasets["nq"]:
#                 logger.info(f"Processed '{split}' size: {len(datasets['nq'][split])}")
#                 logger.info(f"Processed columns: {datasets['nq'][split].column_names}")
#         
#         logger.info("Natural Questions dataset check: SUCCESS")
#         return True
#     except Exception as e:
#         logger.error(f"Error checking Natural Questions dataset: {e}")
#         traceback.print_exc()
#         logger.info("Natural Questions dataset check: FAILED")
#         return False

def check_truthfulqa():
    """Check TruthfulQA dataset."""
    results = {
        "original_format": False,
        "new_binary_format": False,
        "processed_successfully": False
    }
    
    logger.info("======== Checking TruthfulQA Dataset ========")
    
    # First check the original HF format
    try:
        logger.info("Loading TruthfulQA dataset (original multiple choice format)...")
        from datasets import load_dataset
        
        # Load from HF
        truthfulqa_dataset = load_dataset("truthful_qa", "multiple_choice")
        
        # Check the structure
        logger.info(f"Dataset splits: {list(truthfulqa_dataset.keys())}")
        
        if "validation" in truthfulqa_dataset:
            validation_split = truthfulqa_dataset["validation"]
            logger.info(f"Split 'validation' size: {len(validation_split)}")
            logger.info(f"Features: {validation_split.column_names}")
            
            # Check a sample
            if len(validation_split) > 0:
                sample = validation_split[0]
                logger.info(f"Sample keys: {list(sample.keys())}")
                logger.info(f"Sample question: {sample['question']}")
                
                # Check if the correct_answers field exists
                if "correct_answers" in sample:
                    logger.info(f"Sample correct_answers: {sample['correct_answers']}")
                else:
                    logger.info("correct_answers field missing in sample")
                
                # Check MC1 format
                if "mc1_targets" in sample:
                    logger.info(f"Sample mc1_targets: {type(sample['mc1_targets'])} - {sample['mc1_targets']}")
        
        results["original_format"] = True
        logger.info("Original format check: SUCCESS")
    except Exception as e:
        logger.error(f"Error checking TruthfulQA original format: {e}")
        results["original_format"] = False
        logger.info("Original format check: FAILED")
    
    # Now check the new binary choice format
    try:
        logger.info("Checking for updated TruthfulQA binary choice format...")
        
        # Try to load from the local CSV file
        csv_path = os.path.join("dataset", "TruthfulQA.csv")
        logger.info(f"Loading TruthfulQA from local CSV file: {csv_path}")
        
        if os.path.exists(csv_path):
            from datasets import load_dataset
            csv_dataset = load_dataset("csv", data_files={"train": csv_path})
            
            # Check basic structure
            logger.info(f"CSV dataset splits: {list(csv_dataset.keys())}")
            logger.info(f"Split 'train' size: {len(csv_dataset['train'])}")
            logger.info(f"Features: {csv_dataset['train'].column_names}")
            
            # Check a sample
            if "train" in csv_dataset and len(csv_dataset["train"]) > 0:
                sample = csv_dataset["train"][0]
                logger.info(f"CSV sample keys: {list(sample.keys())}")
                
                # Check if it has the binary choice format fields
                if "Best Answer" in sample and "Best Incorrect Answer" in sample:
                    logger.info("Found new binary choice format!")
                    logger.info(f"Best Answer: {sample['Best Answer']}")
                    logger.info(f"Best Incorrect Answer: {sample['Best Incorrect Answer']}")
                    results["new_binary_format"] = True
                else:
                    logger.info("CSV file doesn't have expected binary choice format")
                    results["new_binary_format"] = False
            
            # Now test our processing functions
            logger.info("Processing with our updated data functions...")
            from utils.data_utils import load_and_prepare_datasets
            from config.data_config import DATASETS
            
            datasets = load_and_prepare_datasets({"truthfulqa": DATASETS["truthfulqa"]}, max_samples=2)
            
            if "truthfulqa" in datasets:
                for split in datasets["truthfulqa"]:
                    logger.info(f"Processed '{split}' size: {len(datasets['truthfulqa'][split])}")
                    logger.info(f"Processed columns: {datasets['truthfulqa'][split].column_names}")
                    
                    # Check if it has the binary choice format fields
                    sample = datasets["truthfulqa"][split][0] if len(datasets["truthfulqa"][split]) > 0 else None
                    if sample and "choices" in sample and "labels" in sample:
                        logger.info("Successfully processed with binary choice format!")
                        logger.info(f"Sample choices: {sample['choices']}")
                        logger.info(f"Sample labels: {sample['labels']}")
                        results["processed_successfully"] = True
                    else:
                        logger.info("Processing didn't result in binary choice format")
        else:
            logger.info(f"CSV file not found at: {csv_path}")
            results["new_binary_format"] = False
    except Exception as e:
        logger.error(f"Error checking TruthfulQA binary choice format: {e}")
        logger.error(traceback.format_exc())
        results["new_binary_format"] = False
        results["processed_successfully"] = False
    
    logger.info(f"TruthfulQA check results: {results}")
    
    # Return success if either original format or new format was processed successfully
    return results["original_format"] or results["processed_successfully"]

def check_halueval_qa():
    """Check HaluEval QA dataset."""
    logger.info("======== Checking HaluEval QA Dataset ========")
    try:
        # Load the downloaded JSON file
        logger.info("Loading HaluEval QA dataset...")
        import json
        import os
        from utils.data_utils import load_and_prepare_datasets
        from config.data_config import DATASETS
        
        halueval_path = os.path.join("dataset", "HaluEval", "qa_data.json")
        if not os.path.exists(halueval_path):
            logger.error(f"HaluEval QA dataset file not found at: {halueval_path}")
            logger.info("HaluEval QA dataset check: FAILED")
            return False
            
        # Read a few samples
        with open(halueval_path, "r") as f:
            samples = []
            for i, line in enumerate(f):
                samples.append(json.loads(line))
                if i >= 5:
                    break
        
        # Check the structure
        logger.info(f"Number of samples read: {len(samples)}")
        logger.info(f"Sample keys: {list(samples[0].keys())}")
        logger.info(f"Sample question: {samples[0]['question']}")
        logger.info(f"Sample knowledge/context: {samples[0]['knowledge'][:100]}...")
        logger.info(f"Sample right_answer: {samples[0]['right_answer']}")
        logger.info(f"Sample hallucinated_answer: {samples[0]['hallucinated_answer']}")
        
        # Now test our processing functions by loading
        datasets = load_and_prepare_datasets({"halueval_qa": DATASETS["halueval_qa"]}, max_samples=2)
        
        if "halueval_qa" in datasets:
            for split in datasets["halueval_qa"]:
                logger.info(f"Processed '{split}' size: {len(datasets['halueval_qa'][split])}")
                logger.info(f"Processed columns: {datasets['halueval_qa'][split].column_names}")
                
                # Print a sample
                sample = datasets["halueval_qa"][split][0]
                logger.info(f"Processed sample question: {sample['question']}")
                logger.info(f"Processed sample context: {sample['context'][:100]}...")
                logger.info(f"Processed sample answer: {sample['answer']}")
        
        logger.info("HaluEval QA dataset check: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"Error checking HaluEval QA dataset: {e}")
        traceback.print_exc()
        logger.info("HaluEval QA dataset check: FAILED")
        return False

def main():
    """Main function to run comprehensive checks on all datasets."""
    logger.info("Starting comprehensive dataset checks...")
    
    # Create a timestamp for the run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger.info(f"Timestamp: {timestamp}")
    
    # Run checks on each dataset
    results = {}
    
    logger.info("Checking individual datasets...")
    results["coqa"] = check_coqa()
    results["squad_v2"] = check_squad_v2()
    results["triviaqa"] = check_triviaqa()
    # results["nq"] = check_nq()  # Natural Questions dataset is too large - causing disk quota issues
    results["truthfulqa"] = check_truthfulqa()
    results["halueval_qa"] = check_halueval_qa()
    
    # Print summary
    logger.info("\n===== Dataset Check Summary =====")
    all_passed = True
    for dataset, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{dataset}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll dataset checks passed! 🎉")
        logger.info("The evaluation pipeline is ready to run.")
        return 0
    else:
        logger.error("\nSome dataset checks failed. 😢")
        logger.error("Please fix the issues before running the evaluation pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 