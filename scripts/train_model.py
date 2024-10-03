import torch
from transformers import Trainer, TrainingArguments, GPT2TokenizerFast, GPT2LMHeadModel
from tqdm.auto import tqdm

# Ensure that the model uses the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model, and move model to GPU
tokenizer = GPT2TokenizerFast.from_pretrained('../model/joonGPT')
model = GPT2LMHeadModel.from_pretrained('../model/joonGPT').to(device)

# Load data
with open('../data/formatted_recipes.txt', 'r') as f:
    data = f.readlines()

# Split data
train_size = 0.85
train_len = int(train_size * len(data))
train_data = data[:train_len]
val_data = data[train_len:]

# Define RecipeDataset
class RecipeDataset:
    def __init__(self, data):
        self.data = data
        self.input_ids = []
        self.attn_masks = []
        for item in tqdm(data):
            encodings = tokenizer.encode_plus(item, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')
            self.input_ids.append(torch.squeeze(encodings['input_ids'], 0))  # No need to move to GPU here
            self.attn_masks.append(torch.squeeze(encodings['attention_mask'], 0))  # No need to move to GPU here
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# Create datasets
train_ds = RecipeDataset(train_data)
val_ds = RecipeDataset(val_data)

# Collate function
def collate_fn(batch):
    return {
        'input_ids': torch.stack([item[0] for item in batch]),
        'attention_mask': torch.stack([item[1] for item in batch]),
        'labels': torch.stack([item[0] for item in batch])  # Labels are the same as input_ids for language modeling
    }

# Training arguments
args = TrainingArguments(
    output_dir='../output',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    report_to='none',
    num_train_epochs=3,
    save_strategy='no',
    fp16=True,  # Enable mixed-precision training to speed up on GPU
)

# Optimizer and scheduler
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 20, eta_min=1e-7)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    optimizers=(optim, scheduler)
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('../model/fine_tuned_joonGPT')
tokenizer.save_pretrained('../model/fine_tuned_joonGPT')  # Save the tokenizer

