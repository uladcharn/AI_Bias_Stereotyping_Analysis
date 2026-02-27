import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from textblob import TextBlob

torch.random.manual_seed(0)
torch.cuda.manual_seed_all(0)

class Phi35mini:
    def __init__(self, system_prompt, n_iter = 100, temp = 0.2, max_tokens = 500, return_full_text = False):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct", 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

        self.user_prompt = system_prompt
        self.n_iter = n_iter
        self.temp = temp
        self.max_tokens = max_tokens
        self.return_full_text = return_full_text

    def get_answer(self, messages):
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

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
    
class Gemma3_4B:
    def __init__(self, system_prompt, n_iter=100, temp=0.2, max_tokens=500, return_full_text=False):
        # Note: Gemma 3 is a gated model, ensure you are logged in via huggingface-cli
        self.model_id = "google/gemma-3-4b-it" 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu", # Change to "cuda" if GPU is available
            torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        self.user_prompt = system_prompt
        self.n_iter = n_iter
        self.temp = temp
        self.max_tokens = max_tokens
        self.return_full_text = return_full_text

    def get_answer(self, messages):
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

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
        # Gemma 3 expects this specific chat template format
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
    def __init__(self, system_prompt, n_iter=100, temp=0.2, max_tokens=500, return_full_text=False):
        # Ensure you have accepted the license on Hugging Face
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu",
            torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Llama 3.2 needs a pad_token defined if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.user_prompt = system_prompt
        self.n_iter = n_iter
        self.temp = temp
        self.max_tokens = max_tokens
        self.return_full_text = return_full_text

    def get_answer(self, messages):
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

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