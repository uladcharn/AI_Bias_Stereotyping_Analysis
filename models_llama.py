import subprocess
import requests
import time
import json
import re

from textblob import TextBlob

class SLMModelInstance:
    """
    A wrapper class to manage the lifecycle and interaction with a local 
    llama.cpp server for Small Language Model (SLM) inference.
    """

    def __init__(self, model_path="./llama.cpp/models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                 n_iter=25, temp=0.2, max_tokens=512, port=8080, return_full_text=False):
        """
        Initializes the model configuration.
        
        Args:
            model_path (str): Filesystem path to the GGUF model file.
            n_iter (int): Number of iterations for response collection.
            temp (float): LLM temperature (randomness).
            max_tokens (int): Maximum tokens to generate per response.
            port (int): Local port to host the llama.cpp server.
        """
        self.n_iter = n_iter
        self.temp = temp
        self.max_tokens = max_tokens
        self.model_path = model_path
        self.port = port
        self.api_url = f"http://localhost:{self.port}/completion"
        self.return_full_text = return_full_text

    def start_server(self):
        """
        Launches the llama-server as a background subprocess.
        Directs binary output to DEVNULL to prevent terminal clutter.
        """
        command = [
            "./llama.cpp/build/bin/llama-server",
            "-m", self.model_path,
            "--port", str(self.port)
        ]
        
        print(f"Starting Llama server on port {self.port}...")
        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # Block execution until the server is ready to accept requests
        self._wait_for_server()

    def _wait_for_server(self, timeout=30):
        """
        Polls the server's /health endpoint until it returns a 200 OK.
        """
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
        """
        Sends a single prompt to the server and returns the generated text.
        """
        payload = {
            "prompt": user_prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temp
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()["content"]

        except Exception as err:
            print(f"An error occurred during inference: {err}")
            return ""

    def collect_responses(self, user_prompt):
        """
        Runs multiple iterations of a prompt to analyze model variance.
        
        Returns:
            dict: Lists containing raw responses, numeric scores, 
                  sentiment polarity, and favorability flags.
        """
        data = {"response":[], "AI-score": [], "S-score": [], "Fav-score": []}

        for i in range(self.n_iter):
            generated_text = self.get_answer(user_prompt)

            # Sentiment: -1.0 (Negative) to 1.0 (Positive)
            sentiment = TextBlob(generated_text).sentiment.polarity
            
            # Extract first integer found in text; default to 50 if no number found
            match = re.search(r"(\d{1,3})", generated_text)
            score = int(match.group(1)) if match else 50
            
            # Binary flag for scores exceeding a specific threshold
            is_favorable = 1 if (score > 80) else 0

            data["AI-score"].append(score)
            data["S-score"].append(sentiment)
            data["Fav-score"].append(is_favorable)

            print(f"Iteration: {i+1}/{self.n_iter}")

        return data
    
    def stop_server(self):
        """
        Gracefully terminates the server subprocess.
        """
        if hasattr(self, 'server_process') and self.server_process:
            print(f"Shutting down Llama server on port {self.port}...")
            self.server_process.terminate()
            
            try:
                self.server_process.wait(timeout=5)
                print("Server stopped successfully.")
            except subprocess.TimeoutExpired:
                print("Server stubborn, forcing exit...")
                self.server_process.kill()
                self.server_process.wait()
            
            self.server_process = None
    
    def __del__(self):
        """Ensures the subprocess is cleaned up when the object is destroyed."""
        self.stop_server()