from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Create FastAPI app with metadata for Swagger
app = FastAPI(
    title="Code Generation API",
    description="This API uses a fine-tuned LLM for code generation. Use the /generate endpoint to generate code from a prompt.",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI will be available at /docs
    redoc_url="/redoc",     # ReDoc documentation will be available at /redoc
    openapi_url="/openapi.json"
)

# Define the path to your fine-tuned model
model_path = "models/fine_tuned"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure the tokenizer has a padding token (use eos_token as fallback)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

@app.post("/generate")
async def generate_code(prompt: str):
    """
    Generate code based on a provided prompt.
    - **prompt**: A string with your code prompt.
    """
    # Tokenize the prompt and move the inputs to the appropriate device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate code with sampling parameters
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
