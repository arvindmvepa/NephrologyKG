from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your_dataset_name")
# Split the dataset into training and validation sets
train_dataset = dataset["train"]
eval_dataset = dataset["test"]  # or dataset["validation"], depending on your dataset

# Load the tokenizer and model
model_name = "your_pretrained_zephyr_model_name_here"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=your_num_labels)

# Tokenize the input texts
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory for model checkpoints
    evaluation_strategy="epoch",     # evaluate each `logging_steps`
    learning_rate=2e-5,              # initial learning rate
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # total number of training epochs
    weight_decay=0.01,               # strength of weight decay
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./your_fine_tuned_model_directory")
tokenizer.save_pretrained("./your_fine_tuned_model_directory")