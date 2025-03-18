#!/usr/bin/env python3
"""
Simple test script to directly load and examine the CoQA dataset structure.
"""

import os
import sys
import logging
import argparse
from datasets import load_dataset, Dataset

# Set up logging to both console and file
LOG_FILE = "coqa_test_results.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    logger.info("Loading CoQA dataset...")
    
    try:
        # Load the dataset directly
        ds = load_dataset("stanfordnlp/coqa")
        
        # Examine dataset structure
        logger.info("Dataset splits: %s", ds.keys())
        
        for split_name in ds.keys():
            split = ds[split_name]
            logger.info("Split: %s", split_name)
            logger.info("  Size: %d", len(split))
            logger.info("  Features: %s", split.features)
            logger.info("  Column names: %s", split.column_names)
            
            # Look at the first example
            if len(split) > 0:
                first_example = split[0]
                logger.info("  First example type: %s", type(first_example))
                
                # Examine the structure of the first example
                if isinstance(first_example, dict):
                    logger.info("  First example keys: %s", list(first_example.keys()))
                    
                    # Examine specific fields
                    for key in first_example.keys():
                        value = first_example[key]
                        logger.info("    %s: %s (type: %s)", key, str(value)[:100], type(value))
                        
                        # If it's a list or dict, show more details
                        if isinstance(value, (list, dict)) and len(value) > 0:
                            logger.info("      Length: %d", len(value))
                            if isinstance(value, dict):
                                logger.info("      Keys: %s", list(value.keys()))
                            elif isinstance(value, list):
                                logger.info("      First item: %s (type: %s)", 
                                          str(value[0])[:50], type(value[0]))
                
        # Process a few examples to see their structure
        logger.info("\nProcessing examples...")
        
        def process_coqa_example(example):
            """Process a single CoQA example for QA format."""
            questions = example["questions"]
            answers = example["answers"]
            context = example["story"]
            
            # Extract the first QA pair for simplicity
            processed = {
                "question": questions[0] if questions else "",
                "answer": answers["input_text"][0] if "input_text" in answers and answers["input_text"] else "",
                "context": context
            }
            
            return processed
        
        # Process a few examples from the validation set
        val_examples = ds["validation"].select(range(min(3, len(ds["validation"]))))
        
        for i, example in enumerate(val_examples):
            logger.info("Example %d:", i)
            processed = process_coqa_example(example)
            logger.info("  Original keys: %s", list(example.keys()))
            logger.info("  Processed keys: %s", list(processed.keys()))
            logger.info("  Question: %s", processed["question"])
            logger.info("  Answer: %s", processed["answer"])
            logger.info("  Context (first 100 chars): %s", processed["context"][:100])
        
        # Create a dataset from processed examples
        processed_examples = val_examples.map(process_coqa_example)
        logger.info("\nProcessed dataset:")
        logger.info("  Size: %d", len(processed_examples))
        logger.info("  Column names: %s", processed_examples.column_names)
        
        # Convert to a standardized dictionary format and then to Dataset
        processed_dict = {
            "question": processed_examples["question"],
            "answer": processed_examples["answer"],
            "context": processed_examples["context"]
        }
        
        final_dataset = Dataset.from_dict(processed_dict)
        logger.info("\nFinal dataset:")
        logger.info("  Size: %d", len(final_dataset))
        logger.info("  Column names: %s", final_dataset.column_names)
        
        # Test accessing elements from the final dataset
        logger.info("\nAccessing elements from final dataset:")
        for i in range(min(3, len(final_dataset))):
            item = final_dataset[i]
            logger.info("  Item %d type: %s", i, type(item))
            logger.info("  Item %d: %s", i, item)
            logger.info("  Question: %s", item["question"])
            logger.info("  Answer: %s", item["answer"])
        
        logger.info("\nTest completed successfully!")
        return 0
        
    except Exception as e:
        logger.error("Error: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 