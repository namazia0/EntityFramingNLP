from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from dotenv import load_dotenv
import torch
import os

load_dotenv(dotenv_path="config/config.env")

# Load base model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Directory containing text files
language = os.getenv("LANGUAGE")  # EN or PT (Portuguese)
data_dir = f"./dataset/train_4_december/{language}/raw-documents/"
entity_path = f"./dataset/train_4_december/{language}/subtask-1-annotations.txt"

dataset_list = []
true_labels = []  # Store true labels separately before tokenization

# Read annotation file
with open(entity_path, "r", encoding="utf-8") as file:
    for line in file:
        elements = line.strip().split("\t")
        filename = elements[0]
        file_path = os.path.join(data_dir, filename)

        # Extract text from the file
        with open(file_path, "r", encoding="utf-8") as text_file:
            full_text = text_file.read().strip()

        entity = elements[1]
        start_offset, end_offset = int(elements[2]), int(elements[3])
        main_role = elements[4]

        # Handle optional sub-roles
        if len(elements) == 6:
            sub_role = elements[5]
        elif len(elements) == 7:
            sub_role = f"{elements[5]}, {elements[6]}"

        # Create structured output format
        output_label = f"Main role: {main_role}, Sub roles: {sub_role}"
        true_labels.append(output_label)  # Store true label

        # Create dataset entry
        dataset_list.append({
            "input": f"Text: {full_text}\nEntity: {entity}\nOffset: ({start_offset}, {end_offset})",
            "output": output_label  # Explicitly define "output"
        })

print("Total dataset size:", len(dataset_list))

# Convert list to Dataset object
dataset = Dataset.from_list(dataset_list)

# Tokenization function
def preprocess(example):
    inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    labels = tokenizer(
        example["output"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    inputs["labels"] = labels["input_ids"]  # Assign tokenized labels
    return inputs

# Apply tokenization to dataset
dataset = dataset.map(preprocess, batched=True, remove_columns=["input", "output"])

# Split dataset into train and test
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Adjust `true_labels` to match test dataset size
test_true_labels = true_labels[-len(eval_dataset):]  # Get last N labels for evaluation

# Define LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_fine_tuned_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    remove_unused_columns=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train the LoRA-adapted model
trainer.train()

# Save
model.save_pretrained("./lora_model")

# ===============================
#  GENERATE PREDICTIONS & EVALUATE
# ===============================

print("Generating predictions on the test set...")

# Get model predictions
predictions = trainer.predict(eval_dataset)

token_ids = torch.argmax(torch.tensor(predictions.predictions), dim=-1)

# Decode token IDs to text
decoded_preds = tokenizer.batch_decode(token_ids, skip_special_tokens=True)

test_true_labels = test_true_labels[:len(decoded_preds)]

correct = 0

print("\nPredictions vs. True Labels:\n")
for i, (pred, true) in enumerate(zip(decoded_preds, test_true_labels)):
    pred = pred.strip()
    true = true.strip()
    is_correct = "True" if pred == true else "False"
    
    print(f"Example {i+1}:")
    print(f"  Prediction: {pred}")
    print(f"  True Label: {true}")
    print(f"  Correct? {is_correct}\n")
    
    if pred == true:
        correct += 1

# Compute accuracy
accuracy = correct / len(test_true_labels)

print(f"\nFinal Accuracy on Test Set: {accuracy:.4f} ({correct}/{len(test_true_labels)})")
