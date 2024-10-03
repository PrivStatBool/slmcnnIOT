import pandas as pd

# Load the dataset from the correct path
clean = pd.read_csv('../data/recipes.csv')

# Print the column names to check if 'ingredients' and 'instructions' exist
print("Column names in the dataset:")
print(clean.columns)

# Shuffle the dataset
clean = clean.sample(frac=1)

# Reset the index after shuffling
clean.reset_index(drop=True, inplace=True)

# Function to print a recipe at a given index (for testing purposes)
def print_recipe(idx):
    print(f"Ingredients:\n{clean['Cleaned-Ingredients'][idx]}\n\nInstructions:\n{clean['TranslatedInstructions'][idx]}")

# Function to format a recipe string
def form_string(ingredient, instruction):
    s = f"<|startoftext|>Ingredients:\n{ingredient.strip()}\n\nInstructions:\n{instruction.strip()}<|endoftext|>"
    return s

# Apply the form_string function to the entire dataset
data = clean.apply(lambda x: form_string(x['Cleaned-Ingredients'], x['TranslatedInstructions']), axis=1).to_list()

# Save the formatted data to a text file
with open('../data/formatted_recipes.txt', 'w') as f:
    for item in data:
        f.write("%s\n" % item)

# Print the first three formatted recipes to verify cleansing
for i in range(3):
    print(f"Formatted Recipe {i+1}:\n{data[i]}\n")

