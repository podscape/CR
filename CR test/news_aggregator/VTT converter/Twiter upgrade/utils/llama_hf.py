# utils/llama_hf.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AsyncTextGenerator:
    def __init__(self, model_name: str):
        """Initialize the LLaMA model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")

    async def generate_text(self, prompt: str) -> 'Response':
        """Generate text from prompt asynchronously"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate text
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
            # Decode and return response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return Response(response_text)
            
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return Response("")

class Response:
    """Simple response wrapper"""
    def __init__(self, text: str):
        self.response = text