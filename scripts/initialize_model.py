import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# Define model and tokenizer
model_name = 'gpt2'
model_save_path = '../model/joonGPT'

# Load the tokenizer with custom tokens
tokenizer = GPT2TokenizerFast.from_pretrained(model_name,
                                              bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>',
                                              unk_token='<|unknown|>',
                                              pad_token='<|pad|>'
                                             )

# Add the special tokens to the tokenizer (optional, in case you need to verify)
special_tokens = ['<|startoftext|>', '<|endoftext|>', '<|pad|>', '<|unknown|>']
special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
print(f"Special Token IDs: {special_token_ids}")

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Resize token embeddings to account for new tokens
model.resize_token_embeddings(len(tokenizer))

# Check the new embedding size
new_embedding_size = model.transformer.wte.num_embeddings
print(f"New embedding size (vocabulary size): {new_embedding_size}")

# Save model and tokenizer to the defined path
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Convert '<|pad|>' token to its ID and print it for verification
pad_token_id = tokenizer.convert_tokens_to_ids(['<|pad|>'])
print(f"Token ID for <|pad|>: {pad_token_id}")  # This should print [50259] or the respective ID

# Print model and tokenizer save confirmation
print(f"Model and tokenizer saved to {model_save_path}")

# Add a function to generate text from a prompt for testing
#def generate(prompt):
#    inputs = tokenizer.encode_plus(prompt, return_tensors='pt')
#    output = model.generate(**inputs, max_length=256, do_sample=True, pad_token_id=pad_token_id[0])
#    print(tokenizer.decode(output[0]))

def generate(prompt):
    inputs = tokenizer.encode_plus(prompt, return_tensors='pt')
    
    # Control the temperature and top_k for better generation
    output = model.generate(
        **inputs,
        max_length=256,        # Control the length
        do_sample=True,        # Allow for creative generation
        pad_token_id=tokenizer.pad_token_id,  # Use the correct pad token ID
        top_k=50,              # Restrict sampling to top 50 likely words (helps generate more relevant output)
        temperature=0.7        # Introduce randomness to the sampling process
    )
    
    # Decode the output and print
    print(tokenizer.decode(output[0], skip_special_tokens=True))


# Example: Generate text based on a prompt (for testing)
generate("Ingredients:\nsalt, oil, chicken")

