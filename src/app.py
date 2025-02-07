from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Load the fine-tuned model (update path when available)
model = AutoModelForCausalLM.from_pretrained("models/fine_tuned/")
tokenizer = AutoTokenizer.from_pretrained("models/fine_tuned/")

@app.post("/generate")
def generate_code(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_code": generated_code}
