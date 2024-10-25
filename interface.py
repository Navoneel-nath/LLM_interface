import os
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    BertTokenizer,
    RobertaTokenizer
)
import torch
from huggingface_hub import login

def authenticate_hugging_face(token):
    login(token)

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

tokenizer_mapping = {
    "gpt2": GPT2Tokenizer,
    "bert": BertTokenizer,
    "roberta": RobertaTokenizer
}

def load_model_and_tokenizer(model_name_or_path):
    model_path = os.path.join(model_dir, model_name_or_path.replace("/", "_"))

    if not os.path.exists(model_path):
        print(f"Downloading model and tokenizer for '{model_name_or_path}'...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=model_path, device_map="auto")
          
            model_type = model.config.model_type 
            tokenizer_class = tokenizer_mapping.get(model_type, AutoTokenizer)
            tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=model_path)
            print(f"Model and tokenizer downloaded and saved to {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None, None
    else:
        print(f"Loading model and tokenizer from local directory '{model_path}'...")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
      
        model_type = model.config.model_type
        tokenizer_class = tokenizer_mapping.get(model_type, AutoTokenizer)
        tokenizer = tokenizer_class.from_pretrained(model_path)

    return model, tokenizer

def load_model_from_input(model_name_or_path):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    if model is None or tokenizer is None:
        return "Error loading model. Please check the model name."
    return "Model loaded successfully!"

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def gradio_interface(model_name, prompt):
    load_message = load_model_from_input(model_name)
    if "Error" in load_message:
        return load_message

    return generate_response(prompt)

if __name__ == "__main__":
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="Hugging Face Model Name", placeholder="Enter model name (e.g., 'gpt2')"),
            gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
        ],
        outputs="text",
        title="Chatbot",
        description="Talk to the LLM! Enter a Hugging Face model name and a prompt."
    )
    iface.launch()
