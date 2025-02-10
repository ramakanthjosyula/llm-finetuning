from fastapi import FastAPI, Form
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import os

app = FastAPI(
    title="Code Generation API",
    description="This API allows for downloading models, fine-tuning them, and performing inference.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Dictionary to store model instances for inference
model_cache = {}

# Directory where models are stored
MODEL_DIR = Path("models")

# Create the models directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Download and fine-tune a model dynamically
def download_model(model_name: str):
    model_dir = MODEL_DIR / model_name

    # Check if model already exists
    if not model_dir.exists():
        print(f"Downloading model {model_name}...")
        # Download the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create directory to save the model
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model {model_name} saved in {model_dir}")
    else:
        print(f"Model {model_name} already exists, skipping download.")

    return model_dir

# Fine-tune a model with the given name and dataset
@app.post("/finetune")
async def finetune_model(model_name: str, data_path: str):
    """
    Fine-tune a model with the given name/path and data.
    - **model_name**: The model name (or path) to fine-tune.
    - **data_path**: The path to the dataset used for fine-tuning.
    """
    # Download or load the model
    model_dir = download_model(model_name)

    # Load the model and tokenizer from the directory
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Set padding token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Fine-tune the model (simplified for illustration)
    print(f"Fine-tuning {model_name} on data from {data_path}...")
    # Add fine-tuning logic here based on the data path

    return {"message": f"Model {model_name} fine-tuned successfully!"}

# GET endpoint to list available models
@app.get("/models")
async def list_models():
    """
    Get the list of available models for inference.
    """
    model_dirs = [d.name for d in MODEL_DIR.iterdir() if d.is_dir()]
    return {"available_models": model_dirs}

@app.post("/generate")
async def generate_code(model_name: str = None, prompt: str = Form(...)):
    """
    Generate code based on a provided prompt and model name.
    - **model_name**: The model name to use for inference. If not provided, the first available model will be used.
    - **prompt**: The input prompt for code generation.
    """
    # If no model_name is provided, use the first available model
    if model_name is None:
        model_name = next(iter(os.listdir(MODEL_DIR)), None)  # Get the first available model

    if model_name is None:
        return {"error": "No models available. Please fine-tune a model first."}

    model_dir = MODEL_DIR / model_name
    if not model_dir.exists():
        return {"error": f"Model {model_name} not found. Please fine-tune the model first."}

    # Load the model and tokenizer from the directory
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize the prompt and move the input tensors to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate code with the provided model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )

    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_code": generated_code}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Code Generation API. Visit /docs for the Swagger UI."}

