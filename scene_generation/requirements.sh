# yes | conda create -n cse252d python=3.10
# conda activate cse252d
pip install uv
uv pip install torch torchvision torchaudio
uv pip install -U "autogen-agentchat" "autogen-ext[openai]"
uv pip install transformers datasets diffusers peft accelerate crewai ipykernel sentencepiece gradio
pip install qwen-vl-utils
# uv pip install flash-attn --no-build-isolation
