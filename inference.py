#!/usr/bin/env python3
"""
Inference script for using hallucination-mitigated models.
"""

import os
import argparse
import logging
from typing import Dict, Any, List, Optional
import torch
from transformers import TextStreamer

from config.model_config import MODELS
from utils.model_utils import load_base_model_and_tokenizer, load_finetuned_model


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a hallucination-mitigated model")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b",
        help="Name of the base model (from config)",
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter. If not provided, will use the base model.",
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
        "--prompt",
        type=str,
        default=None,
        help="Prompt to use for generation. If not provided, will enter interactive mode.",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of generated text",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top p for nucleus sampling",
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save generation output (optional)",
    )
    
    return parser.parse_args()


def format_prompt(prompt: str) -> str:
    """
    Format a prompt for the model.
    
    Args:
        prompt: Raw prompt
        
    Returns:
        Formatted prompt
    """
    # Format prompt using the Llama chat template
    return f"[INST] {prompt} [/INST]"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response from the model.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        prompt: Prompt to generate from
        max_length: Maximum length of generated text
        temperature: Temperature for generation
        top_p: Top p for nucleus sampling
        
    Returns:
        Generated text
    """
    # Format prompt
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Set up text streamer for real-time output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            streamer=streamer,
        )
    
    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the model's response after the instruction
    response = output_text.split("[/INST]")[-1].strip()
    
    return response


def interactive_mode(
    model,
    tokenizer,
    args,
) -> None:
    """
    Run the model in interactive mode.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        args: Command line arguments
    """
    print("\n" + "=" * 50)
    print("Interactive mode. Type 'exit' or 'quit' to exit.")
    print("=" * 50 + "\n")
    
    history = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting interactive mode.")
            break
        
        # Generate response
        print("\nModel: ", end="")
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=user_input,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        # Save to history
        history.append((user_input, response))
    
    # Save history if output file is specified
    if args.output_file and history:
        with open(args.output_file, "w") as f:
            for i, (prompt, response) in enumerate(history):
                f.write(f"Conversation {i+1}:\n")
                f.write(f"User: {prompt}\n")
                f.write(f"Model: {response}\n\n")
        
        print(f"Conversation history saved to {args.output_file}")


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    if args.adapter_path:
        # Load fine-tuned model with adapter
        model, tokenizer = load_finetuned_model(
            base_model_name=args.model_name,
            adapter_path=args.adapter_path,
            load_in_8bit=args.load_in_8bit,
        )
    else:
        # Load base model
        model, tokenizer = load_base_model_and_tokenizer(
            model_name=args.model_name,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )
    
    # Make sure model is in evaluation mode
    model.eval()
    
    # Log model information
    logger.info(f"Model: {args.model_name}")
    if args.adapter_path:
        logger.info(f"Adapter: {args.adapter_path}")
    
    # Check if prompt is provided
    if args.prompt:
        # Single generation mode
        logger.info("Generating response...")
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        # Save to file if specified
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(f"Prompt: {args.prompt}\n\n")
                f.write(f"Response: {response}\n")
            
            logger.info(f"Response saved to {args.output_file}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()