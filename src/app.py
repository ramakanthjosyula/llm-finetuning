from fastapi import FastAPI, Form
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

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

# Download and fine-tune a model dynamically
@app.post("/finetune")
async def finetune_model(model_name: str, data_path: str):
    """
    Fine-tune a model with the given name/path and data.
    - **model_name**: The model name (or path) to fine-tune.
    - **data_path**: The path to the dataset used for fine-tuning.
    """
    # Load the model and tokenizer based on the provided model name/path
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Fine-tune the model (simplified for illustration)
    # This is where you would load your dataset and train the model
    print(f"Fine-tuning {model_name} on data from {data_path}...")
    # Add fine-tuning logic here based on the data path

    # Store the fine-tuned model in memory for later inference
    model_cache[model_name] = model

    return {"message": f"Model {model_name} fine-tuned successfully!"}


@app.post("/generate")
async def generate_code(model_name: str, prompt: str):
    """
    Generate code based on a provided prompt and model name.
    - **model_name**: The model name to use for inference.
    - **prompt**: The input prompt for code generation.
    """
    if model_name not in model_cache:
        return {"error": "Model not found. Please fine-tune the model first."}

    # Load the fine-tuned model from cache
    model = model_cache[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the prompt and move the input tensors to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Generate code with the provided model
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.1)

    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_code": generated_code}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Code Generation API. Visit /docs for the Swagger UI."}
