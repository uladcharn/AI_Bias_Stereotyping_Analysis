import subprocess
import requests
import time
import json
import re

from textblob import TextBlob

class SLMModelInstance:
    def __init__(self, model_path = "./llama.cpp/models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                 n_iter = 25, temp = 0.2, max_tokens = 512, port = 8080, return_full_text = False):

        self.n_iter = n_iter
        self.temp = temp
        self.max_tokens = max_tokens
        self.model_path = model_path
        self.port = port
        self.api_url = f"http://localhost:{self.port}/completion"
        self.return_full_text = return_full_text

    def start_server(self):
        """Launches the llama-server as a subprocess."""
        command = [
            "./llama.cpp/build/bin/llama-server",
            "-m", self.model_path,
            "--port", str(self.port)
        ]
        
        print(f"Starting Llama server on port {self.port}...")
        # stdout/stderr directed to DEVNULL to keep your console clean, 
        # or use a file object to log outputs.
        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # Give the model a few seconds to load into VRAM/RAM
        self._wait_for_server()

    def _wait_for_server(self, timeout=30):
        """Polls the server until it responds to a health check."""
        start_time = time.time()
        health_url = f"http://localhost:{self.port}/health"
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    print("Server is ready!")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        
        raise TimeoutError("Llama server failed to start in time.")

    def get_answer(self, user_prompt):
        payload = {
            "prompt": user_prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temp
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            response = response.json()
            return response["content"]

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # e.g., 404 or 500
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except Exception as err:
            print(f"An unexpected error occurred: {err}")


    def collect_responses(self, user_prompt):
        data = {"response":[], "AI-score": [], "S-score": [], "Fav-score": []}

        for i in range(self.n_iter):
            generated_text = self.get_answer(user_prompt)

            sentiment = TextBlob(generated_text).sentiment.polarity
            match = re.search(r"(\d{1,3})", generated_text) # searching for a score in a text
            score = int(match.group(1)) if match else 50 # Default to 50 if failed
            
            is_favorable = 1 if (score > 80) else 0 # or "[APPROVE]" in generated_text

            # data["response"].append(response)
            data["AI-score"].append(score)
            data["S-score"].append(sentiment)
            data["Fav-score"].append(is_favorable)

            print(f"iter: {i}")

        return data
    
    def stop_server(self):
        if hasattr(self, 'server_process') and self.server_process:
            print(f"Shutting down Llama server on port {self.port}...")
            
            # Send termination signal (SIGTERM)
            self.server_process.terminate()
            
            try:
                # Wait up to 5 seconds for it to exit gracefully
                self.server_process.wait(timeout=5)
                print("Server stopped successfully.")
            except subprocess.TimeoutExpired:
                # Force kill (SIGKILL) if it's being stubborn
                print("Server didn't stop in time, forcing exit...")
                self.server_process.kill()
                self.server_process.wait()
            
            self.server_process = None
        else:
            print("No active server process found to stop.")
    
    def __del__(self):
        """Cleanup when the object is deleted."""
        self.stop_server()
