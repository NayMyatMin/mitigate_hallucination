"""
Utility functions for model evaluation, focusing on hallucination metrics.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from config.data_config import EVAL_DATASETS


def calculate_hallucination_rate(
    references: List[str], 
    predictions: List[str],
    factual_keywords: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate the hallucination rate based on factual keywords.
    
    Args:
        references: List of reference answers
        predictions: List of model predictions
        factual_keywords: List of factual keywords to check for
        
    Returns:
        Dictionary with hallucination metrics
    """
    # If no factual keywords provided, extract from references
    if factual_keywords is None:
        factual_keywords = []
        for ref in references:
            # Extract key entities/facts from reference
            # This is a simplified approach - in production you'd use NER or other techniques
            words = re.findall(r'\b[A-Za-z]{4,}\b', ref)
            factual_keywords.extend([w.lower() for w in words if len(w) > 4])
        
        # Get unique keywords
        factual_keywords = list(set(factual_keywords))
    
    # Calculate metrics
    keyword_presence = []
    for pred in predictions:
        pred_lower = pred.lower()
        # Check what percentage of factual keywords are in the prediction
        if factual_keywords:
            present = sum(1 for kw in factual_keywords if kw.lower() in pred_lower)
            keyword_presence.append(present / len(factual_keywords))
        else:
            keyword_presence.append(1.0)  # No keywords to check
    
    # Calculate metrics
    avg_keyword_presence = np.mean(keyword_presence)
    hallucination_score = 1.0 - avg_keyword_presence
    
    return {
        "hallucination_rate": hallucination_score,
        "factual_consistency": avg_keyword_presence,
    }


def check_citation_quality(
    predictions: List[str],
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Evaluate citation quality in generated text.
    
    Args:
        predictions: List of model predictions
        sources: List of dictionaries with source information
        
    Returns:
        Dictionary with citation quality metrics
    """
    # Check for citation patterns in predictions
    citation_pattern = r'(\[[\d,\s]+\]|\.[\d]+|Citation:[\s\d,]+)'
    
    has_citation = [1 if re.search(citation_pattern, pred) else 0 for pred in predictions]
    citation_rate = sum(has_citation) / len(predictions)
    
    # If sources are provided, check if citations match sources
    citation_accuracy = 0.0
    if sources:
        # This would need to be implemented based on your citation format
        # For now, we'll just return the citation rate
        citation_accuracy = citation_rate
    
    return {
        "citation_rate": citation_rate,
        "citation_accuracy": citation_accuracy,
    }


def evaluate_truthfulness_with_truthfulqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    num_samples: int = 100,
    eval_mode: str = "binary_choice",  # 'binary_choice', 'multiple_choice', or 'generation'
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model truthfulness using TruthfulQA benchmark.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        max_length: Maximum generation length
        num_samples: Number of samples to evaluate
        eval_mode: Mode for evaluation ('binary_choice', 'multiple_choice', or 'generation')
        device: Device to run inference on
        
    Returns:
        Dictionary with TruthfulQA metrics
    """
    try:
        # Try to load using the data_utils.py with our updated binary format
        # Check if we have the function available
        from utils.data_utils import load_and_prepare_datasets
        from config.data_config import DATASETS
        
        # Load the dataset with the improved binary choice format
        try:
            datasets = load_and_prepare_datasets({"truthfulqa": DATASETS["truthfulqa"]}, max_samples=num_samples)
            dataset = datasets["truthfulqa"]["eval"]
            print(f"Successfully loaded TruthfulQA with {len(dataset)} samples")
            print(f"Dataset features: {dataset.column_names}")
            has_binary_choices = any(example.get('is_binary_choice', False) for example in dataset)
            if has_binary_choices:
                print("Using new binary choice format for TruthfulQA")
            else:
                print("Using original format for TruthfulQA (binary choices not detected)")
        except Exception as e:
            print(f"Error loading with data_utils: {e}")
            # Fall back to direct loading
            dataset = load_dataset("truthful_qa", "multiple_choice")
            if "validation" in dataset:
                dataset = dataset["validation"]
            else:
                dataset = dataset["test"]
            has_binary_choices = False
            print("Loaded TruthfulQA directly (fallback method)")
        
        # Limit to num_samples
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        # Based on evaluation mode
        if eval_mode == "binary_choice" and has_binary_choices:
            # Use our new binary choice evaluation
            binary_prompt_template = """[INST] Answer the following multiple-choice question. Choose (A) or (B).

Question: {question}

Options:
(A) {option_a}
(B) {option_b} [/INST]"""
            
            return evaluate_truthfulqa_binary_choice(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                binary_prompt_template=binary_prompt_template,
                device=device
            )
        
        elif eval_mode == "multiple_choice" or (eval_mode == "binary_choice" and not has_binary_choices):
            # Original multiple choice evaluation
            # Format questions for multiple choice
            model_scores = []
            total = 0
            correct = 0
            
            for example in dataset:
                # Get MC1 data (single true answer format)
                if 'mc1_labels' in example and 'mc1_targets' in example:
                    choices = example['mc1_labels']
                    targets = example['mc1_targets']
                elif 'choices' in example and 'labels' in example:
                    choices = example['choices']
                    targets = example['labels']
                else:
                    continue
                    
                total += 1
                
                # Compute log probability for each choice
                log_probs = []
                for choice in choices:
                    prompt = f"[INST] {example['question']} [/INST] {choice}"
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        # Get log prob of the choice tokens
                        shift_logits = logits[0, :-1, :].contiguous()
                        shift_labels = inputs.input_ids[0, 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        choice_log_prob = -loss.sum().item()
                        log_probs.append(choice_log_prob)
                
                # Find the choice with highest log prob
                best_idx = log_probs.index(max(log_probs))
                model_choice = best_idx
                
                # Check if correct
                if targets[model_choice] == 1:
                    correct += 1
                    
                model_scores.append({
                    'question': example['question'],
                    'choices': choices,
                    'model_choice': model_choice,
                    'is_correct': targets[model_choice] == 1
                })
            
            mc_accuracy = correct / total if total > 0 else 0.0
            return {
                "truthfulqa_mc_accuracy": mc_accuracy,
                "total_questions": total,
                "results": model_scores
            }
                
        else:  # Generation mode
            # Format questions for the model
            questions = [f"[INST] {example['question']} [/INST]" for example in dataset]
            
            # Generate answers
            model_outputs = []
            for question in questions:
                inputs = tokenizer(question, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        num_return_sequences=1,
                    )
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the model's response after the instruction
                response = output_text.split("[/INST]")[-1].strip()
                model_outputs.append(response)
            
            # Analyze truthfulness
            correct_count = 0
            for i, output in enumerate(model_outputs):
                example = dataset[i]
                # Get correct answers from different formats
                if 'correct_answers' in example:
                    correct_answers = example['correct_answers']
                elif 'answer' in example:
                    correct_answers = [example['answer']]
                else:
                    # Try to extract from MC format
                    correct_answers = []
                    if 'mc1_labels' in example and 'mc1_targets' in example:
                        for label, target in zip(example['mc1_labels'], example['mc1_targets']):
                            if target == 1:
                                correct_answers.append(label)
                    
                output_lower = output.lower()
                for ans in correct_answers:
                    if isinstance(ans, str) and ans.lower() in output_lower:
                        correct_count += 1
                        break
            
            truthfulness_score = correct_count / len(model_outputs)
            return {
                "truthfulqa_generation_score": truthfulness_score,
                "total_questions": len(model_outputs),
                "generated_responses": model_outputs
            }
    
    except Exception as e:
        print(f"Error evaluating with TruthfulQA: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "truthfulqa_score": 0.0,
            "error": str(e)
        }


def evaluate_hallucination(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    references_key: str = "output",
    input_key: str = "input",
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of hallucination in model outputs.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Dataset for evaluation
        references_key: Key for reference answers in dataset
        input_key: Key for input questions in dataset
        max_length: Maximum generation length
        
    Returns:
        Dictionary with hallucination metrics
    """
    if len(eval_dataset) == 0:
        return {
            "hallucination_rate": 0.0,
            "factual_consistency": 0.0,
            "citation_rate": 0.0,
        }
    
    # Format inputs
    inputs = [f"[INST] {example} [/INST]" for example in eval_dataset[input_key]]
    references = eval_dataset[references_key]
    
    # Generate predictions
    predictions = []
    for input_text in inputs:
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=1,
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the model's response after the instruction
        response = output_text.split("[/INST]")[-1].strip()
        predictions.append(response)
    
    # Calculate hallucination metrics
    hallucination_metrics = calculate_hallucination_rate(references, predictions)
    
    # Calculate citation metrics
    citation_metrics = check_citation_quality(predictions)
    
    # Combine metrics
    combined_metrics = {**hallucination_metrics, **citation_metrics}
    
    return combined_metrics


def evaluate_truthfulqa_binary_choice(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    binary_prompt_template: str = "Question: {question}\n\nOptions:\n(A) {option_a}\n(B) {option_b}",
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate a model on the TruthfulQA binary choice task.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        dataset: The dataset containing binary choices
        binary_prompt_template: Template for formatting binary choice questions
        device: Device to run inference on
    
    Returns:
        Dictionary with accuracy metrics
    """
    import random
    import torch
    from tqdm import tqdm
    
    model.eval()
    correct = 0
    total = 0
    
    results = []
    
    for example in tqdm(dataset):
        if not example.get("is_binary_choice", False) or len(example["choices"]) != 2:
            continue
            
        total += 1
        
        # Get the choices and randomize their order
        choices = example["choices"]
        labels = example["labels"]
        
        # Random ordering
        if random.random() > 0.5:
            option_a, option_b = choices
            label_a, label_b = labels
        else:
            option_b, option_a = choices
            label_b, label_a = labels
            
        # Format the prompt
        prompt = binary_prompt_template.format(
            question=example["question"],
            option_a=option_a,
            option_b=option_b
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Calculate log probabilities for both options
        with torch.no_grad():
            # First calculate for "(A)"
            a_token_ids = tokenizer("(A)").input_ids[1:]  # Skip the first token (BOS)
            a_start_idx = len(inputs.input_ids[0]) - len(a_token_ids)
            a_logits = model(**inputs).logits[0, a_start_idx-1:-1]
            a_log_probs = torch.log_softmax(a_logits, dim=-1)
            a_token_log_probs = torch.gather(a_log_probs, 1, torch.tensor([a_token_ids]).to(device).transpose(0, 1))
            a_score = a_token_log_probs.sum().item()
            
            # Then calculate for "(B)"
            b_token_ids = tokenizer("(B)").input_ids[1:]  # Skip the first token (BOS)
            b_start_idx = len(inputs.input_ids[0]) - len(b_token_ids)
            b_logits = model(**inputs).logits[0, b_start_idx-1:-1]
            b_log_probs = torch.log_softmax(b_logits, dim=-1)
            b_token_log_probs = torch.gather(b_log_probs, 1, torch.tensor([b_token_ids]).to(device).transpose(0, 1))
            b_score = b_token_log_probs.sum().item()
        
        # Determine which option the model prefers
        model_choice = 0 if a_score > b_score else 1  # 0 for A, 1 for B
        true_answer = 0 if label_a == 1 else 1  # 0 for A, 1 for B
        
        # Check if the model's choice is correct
        is_correct = model_choice == true_answer
        if is_correct:
            correct += 1
            
        results.append({
            "question": example["question"],
            "option_a": option_a,
            "option_b": option_b,
            "model_choice": "A" if model_choice == 0 else "B",
            "correct_choice": "A" if true_answer == 0 else "B",
            "is_correct": is_correct
        })
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "truthfulqa_binary_accuracy": accuracy,
        "total_questions": total,
        "results": results
    } 