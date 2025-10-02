#!/usr/bin/env python3
"""Debug script to test few-shot IOI evaluation."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bento  # noqa - register custom model types

def test_few_shot_generation(model_path):
    """Test what the model generates for few-shot prompts."""
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a few-shot prompt with new format
    prompt = """Task: Identify who receives the object in each sentence (the indirect object).

Sentence: Friends Juana and Kristi found a mango at the bar. Kristi gave it to Juana
Who receives the object? Juana

Sentence: Then, Yvette and Angie were working at the mountain. Yvette decided to give a banana to Angie
Who receives the object? Angie

Sentence: After Doris and Marsha went to the mountain, Marsha gave a strawberry to Doris
Who receives the object? Doris

Sentence: While Bernadette and Harriet were commuting to the desert, Bernadette gave a watermelon to Harriet
Who receives the object? Harriet

Sentence: Afterwards, Ginger and Bernadette went to the library. Bernadette gave a mango to Ginger
Who receives the object? Ginger

Sentence: Then, Faye and Rosa had a lot of fun at the theater. Faye gave a blackberry to Rosa
Who receives the object?"""
    
    print("\n" + "="*60)
    print("FEW-SHOT PROMPT:")
    print("="*60)
    print(prompt)
    print("="*60)
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to("cuda")
    
    print(f"\nPrompt tokens: {inputs['input_ids'].shape[1]}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("FULL MODEL OUTPUT:")
    print("="*60)
    print(full_output)
    print("="*60)
    
    # Extract only generated part
    prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    generated_only = full_output[prompt_length:]
    
    print("\n" + "="*60)
    print("GENERATED ONLY:")
    print("="*60)
    print(repr(generated_only))
    print("="*60)
    
    # Try different extraction methods
    print("\n" + "="*60)
    print("EXTRACTION ATTEMPTS:")
    print("="*60)
    
    # Method 1: After last "Who receives the object?"
    search_phrase = "Who receives the object?"
    if search_phrase in full_output:
        answer_start = full_output.rfind(search_phrase) + len(search_phrase)
        extracted = full_output[answer_start:].strip()
        first_word = extracted.split()[0] if extracted else ""
        print(f"Method 1 (after last '{search_phrase}'): '{first_word}'")
    
    # Method 2: Just the generated text
    first_word_gen = generated_only.strip().split()[0] if generated_only.strip() else ""
    print(f"Method 2 (first word of generated): '{first_word_gen}'")
    
    # Let's also try a simpler prompt format
    print("\n\n" + "="*60)
    print("TESTING SIMPLER PROMPT FORMAT:")
    print("="*60)
    
    simple_prompt = """Kristi gave it to Juana. The indirect object is: Juana
Yvette decided to give a banana to Angie. The indirect object is: Angie
Marsha gave a strawberry to Doris. The indirect object is: Doris
Bernadette gave a watermelon to Harriet. The indirect object is: Harriet
Bernadette gave a mango to Ginger. The indirect object is: Ginger
Faye gave a blackberry to Rosa. The indirect object is:"""
    
    print(simple_prompt)
    
    inputs = tokenizer(simple_prompt, return_tensors="pt", padding=False).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    generated_only = full_output[prompt_length:]
    
    print(f"\nGenerated: {repr(generated_only)}")
    

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "exp/transformer_340M_mqar_synthetic/hf_checkpoint_10000"
    test_few_shot_generation(model_path)