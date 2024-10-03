from transformers import pipeline

# Ensure that the pipeline uses the GPU (device=0 for the first GPU)
pl = pipeline(task='text-generation', model='../model/fine_tuned_joonGPT', device=0)

# Create prompt function
def create_prompt(ingredients):
    ingredients = ','.join([x.strip().lower() for x in ingredients.split(',')])
    ingredients = ingredients.strip().replace(',', '\n')
    s = f"<|startoftext|>Ingredients:\n{ingredients}\n\nInstructions:\n"
    return s

# Test with a few ingredient sets
ingredients_list = [
    'Rice, Potatoes, Tomatoes, Spinach, red bell peppers',
    'chicken, tomatoes, aloo, jeera, curry powder'
]

for ingredients in ingredients_list:
    prompt = create_prompt(ingredients)
    result = pl(prompt, 
                max_new_tokens=512,  # Increase the maximum number of tokens
                min_length=100,  # Ensure the model generates at least 100 tokens
                top_k=50,  # Increase diversity
                temperature=0.8,  # Slightly increase randomness
                repetition_penalty=1.2,  # Penalize repetition
                pad_token_id=50259)  # Ensure padding is correct
    print(result[0]['generated_text'])

