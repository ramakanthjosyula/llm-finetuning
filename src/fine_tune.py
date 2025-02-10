import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Function to load code samples from a directory (e.g., data/raw)
def load_data_from_directory(directory):
    code_samples = []
    for filepath in glob.glob(os.path.join(directory, '*.py')):
        with open(filepath, 'r') as f:
            code_samples.append(f.read())
    return code_samples

def tokenize_function(examples, tokenizer):
    # Tokenize the text with the desired parameters
    result = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    # Set labels equal to input_ids so the model can compute the loss
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model_name = "Salesforce/codegen-350M-mono"  # Change this if you prefer another model
    # Load pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move the model to GPU
    model = model.to(device)

    # Fix: Set the pad token to the eos_token if not already set
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset from your code samples
    code_samples = load_data_from_directory("data/raw")
    if not code_samples:
        print("No code samples found in data/raw. Please add some code files and try again.")
        return
    dataset = Dataset.from_dict({"text": code_samples})
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Apply LoRA via PEFT for parameter-efficient fine-tuning
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1)
    model = get_peft_model(model, lora_config)

    # Define training arguments (adjust epochs, batch size, etc., as needed)
    training_args = TrainingArguments(
        output_dir="models/fine_tuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        report_to="none"  # Disables wandb and other reporting integrations
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("models/fine_tuned")
    tokenizer.save_pretrained("models/fine_tuned")
    print("Fine-tuning complete. Model saved in models/fine_tuned.")

if __name__ == "__main__":
    main()
