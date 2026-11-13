# Model set used by the benchmarks (name|path). Source with: . bench/models.sh
M=$HOME/models
MODELS=(
  "qwen27b-q8|$M/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf"
  "qwen3-0.6b-q8|$M/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf"
  "llama3.2-1b-q8|$M/gguf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-Q8_0.gguf"
  "gemma3-1b-q8|$M/gguf/gemma-3-1b-it/gemma-3-1b-it-Q8_0.gguf"
  "mistral7b-q4km|$M/gguf/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
  "olmoe-q4km|$M/gguf/OLMoE-1B-7B-0924-Instruct/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
  "nemotron30b-q4km|$M/gguf/NVIDIA-Nemotron-3-Nano-30B-A3B/NVIDIA-Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf"
)
LLAMA_ROOT=$HOME/.local/share/docdictator/models
