# SLM Bias Auditing Framework: Local Deployment & Analysis

This repository contains the infrastructure for auditing geographic and demographic disparities in Small Language Models (SLMs). The framework utilizes `llama.cpp` for high-performance CPU inference on macOS/Unix-based systems, specifically optimized for **GGUF-quantized** models such as Phi-3.5 Mini, Llama 3.2, and Gemma 3.

---

## 🛠 Prerequisites & Environment Setup

Before building the inference engine, ensure your system has the necessary build tools installed via Homebrew.

### 1. Install Homebrew (macOS)
```bash
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"

# Add Homebrew to your shell path
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Install Build Dependencies (if you don't have those)

```bash
# Ensure Xcode Command Line Tools are present
xcode-select --install

# Install essential tools
brew install git cmake
```

### Building the Inference Engine (```llama.cpp```)

To maximize performance on local hardware, we build ```llama.cpp``` from source with the Server module enabled. This allows the models to be accessed via a local REST API, bypassing heavy Python library dependencies for inference.

```bash
# Clone the repository
git clone [https://github.com/ggerganov/llama.cpp.git](https://github.com/ggerganov/llama.cpp.git) 
cd llama.cpp

# Configure and Build
mkdir -p build && cd build
cmake .. -DLLAMA_BUILD_SERVER=ON
cmake --build . --config Release
cd ..
```

### 🚀 Model Deployment (Example with Phi3.5-Mini 4B)

This project utilizes Q4_K_M Quantization (4-bit) to ensure high-speed inference on CPU while maintaining 95%+ of the original model's reasoning and linguistic capabilities.

To start the inference server using a Phi3.5-Mini 4B model, the following command to deploy the model:
```bash
./build/bin/llama-server -m models/Phi-3.5-mini-instruct-Q4_K_M.gguf --port 8080
```