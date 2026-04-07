# NOTE: This code uses HuggingFaces transformers library, which is only recommended to be used 
## only if the computational resources allow you to. For a much faster inference, resort to models_llama.py

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
from textblob import TextBlob

# Ensure reproducibility by seeding random number generators
torch.random.manual_seed(0)
torch.cuda.manual_seed_all(0)

class Phi35mini:
    """
    Interface for Microsoft's Phi-3.5-mini-instruct.
    Designed for efficient, high-quality reasoning in a small parameter space.
    """
    def __init__(self, system_prompt, n_iter = 100, temp = 0.2, max_tokens = 512, return_full_text = False):
        # Load model to CPU by default. Set device_map="auto" for GPU usage.
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct", 
            device_map="cpu", 
            trust_remote_code=False
        ).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

        self.user_prompt = system_prompt
        self.n_iter = n_iter
        self.temp = temp
        self.max_tokens = max_tokens
        self.return_full_text = return_full_text

    def get_answer(self, messages):
        """Generates a single response using the Transformers pipeline."""
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": self.max_tokens,
            "return_full_text": self.return_full_text,
            "temperature": self.temp,
            # If temp is 0, sampling is disabled (greedy decoding)
            "do_sample": False if self.temp == 0 else True,
        }

        output = pipe(messages, **generation_args)
        return output[0]['generated_text']

    def collect_responses(self):
        """Runs n_iter loops to gather data on model behavior and sentiment."""
        data = {"response":[],"S-score": [], "Fav-score": []}
        messages = [{"role": "user", "content": self.user_prompt}]

        for _ in range(self.n_iter):
            response = self.get_answer(messages)

            # Analyze sentiment (-1 to 1) and look for the keyword "Approve"
            sentiment = TextBlob(response).sentiment.polarity
            is_favorable = 1 if "Approve" in response else 0

            data["response"].append(response)
            data["S-score"].append(sentiment)
            data["Fav-score"].append(is_favorable)

        return data
    
class Gemma3_4B:
    """
    Interface for Google's Gemma 3 4B Instruct.
    Requires Hugging Face authentication (huggingface-cli login).
    """
    def __init__(self, system_prompt, n_iter=100, temp=0.2, max_tokens=500, return_full_text=False):
        self.model_id = "google/gemma-3-4b-it" 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu",
            torch_dtype="auto"
        ).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        self.user_prompt = system_prompt
        self.n_iter = n_iter
        self.temp = temp
        self.max_tokens = max_tokens
        self.return_full_text = return_full_text

    def get_answer(self, messages):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        generation_args = {
            "max_new_tokens": self.max_tokens,
            "return_full_text": self.return_full_text,
            "temperature": self.temp,
            "do_sample": False if self.temp == 0 else True,
        }
        output = pipe(messages, **generation_args)
        return output[0]['generated_text']

    def collect_responses(self):
        data = {"response":[],"S-score": [], "Fav-score": []}
        messages = [{"role": "user", "content": self.user_prompt}]

        for _ in range(self.n_iter):
            response = self.get_answer(messages)
            sentiment = TextBlob(response).sentiment.polarity
            is_favorable = 1 if "Approve" in response else 0

            data["response"].append(response)
            data["S-score"].append(sentiment)
            data["Fav-score"].append(is_favorable)
        return data
    
class Llama32_3B:
    """
    Interface for Meta's Llama 3.2 3B Instruct.
    Requires an accepted community license on Hugging Face.
    """
    def __init__(self, system_prompt, n_iter=100, temp=0.2, max_tokens=500, return_full_text=False):
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu",
            torch_dtype="auto"
        ).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Explicitly set pad_token to eos_token to avoid generation errors
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.user_prompt = system_prompt
        self.n_iter = n_iter
        self.temp = temp
        self.max_tokens = max_tokens
        self.return_full_text = return_full_text

    def get_answer(self, messages):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        generation_args = {
            "max_new_tokens": self.max_tokens,
            "return_full_text": self.return_full_text,
            "temperature": self.temp,
            "do_sample": False if self.temp == 0 else True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        output = pipe(messages, **generation_args)
        return output[0]['generated_text']

    def collect_responses(self):
        data = {"responses":[],"S-score": [], "Fav-score": []}
        messages = [{"role": "user", "content": self.user_prompt}]

        for _ in range(self.n_iter):
            response = self.get_answer(messages)
            sentiment = TextBlob(response).sentiment.polarity
            is_favorable = 1 if "Approve" in response else 0

            data["responses"].append(response)
            data["S-score"].append(sentiment)
            data["Fav-score"].append(is_favorable)
        return data