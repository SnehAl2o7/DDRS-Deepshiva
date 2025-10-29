# app.py
import torch
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- 1. Model Loading (Initialization) ---

# We only load the model once when the application starts
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Mistral 7B Model Loaded Successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Raise an error to stop startup if the model fails to load
    raise RuntimeError(f"Failed to load Mistral 7B: {e}")


# --- 2. FastAPI Setup ---
# Use the lifespan context manager to handle startup/shutdown events (optional, but good practice)
app = FastAPI()

# Pydantic model for input validation
class ChatInput(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    system_message: str = "You are a helpful and enthusiastic tourism expert, specializing in giving crisp and detailed recommendations."


@app.get("/")
def home():
    """Simple health check endpoint."""
    return {"status": "ok", "model": "Mistral 7B API is running"}


@app.post("/chat")
async def chat_endpoint(input_data: ChatInput):
    """The main chatbot API endpoint for text generation."""
    try:
        # Prepare messages in Mistral Instruct format
        messages = [
            {"role": "system", "content": input_data.system_message},
            {"role": "user", "content": input_data.prompt}
        ]
        
        # Tokenize input
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True 
        ).to(model.device)

        # Generate text
        outputs = model.generate(
            input_ids,
            max_new_tokens=input_data.max_new_tokens,
            temperature=input_data.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

        # Decode and extract Assistant's response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        start_tag = "[/INST]"
        
        # Extract response after the [/INST] tag and remove the closing </s>
        if start_tag in full_response:
             response_text = full_response.split(start_tag, 1)[-1].strip().replace("</s>", "").strip()
        else:
             response_text = full_response # Fallback

        return {"response": response_text}

    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
