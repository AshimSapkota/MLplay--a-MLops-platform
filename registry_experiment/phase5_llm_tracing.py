import mlflow
import requests
import time

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("llm-prompt-experiments")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:latest"  # update after checking `ollama list`

@mlflow.trace(name="qwen2-generate")
def query_ollama(prompt: str, temperature: float = 0.7) -> str:
    """Wrapped LLM call — MLflow auto-captures input/output/latency"""
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    })
    return response.json()["response"]

# Try a few different prompts — simulating prompt experimentation
prompts = [
    "Explain what a model registry is in one sentence.",
    "Explain what a model registry is in one sentence, for a 10-year-old.",
    "List 3 benefits of MLOps in bullet points.",
]

with mlflow.start_run(run_name="qwen2-prompt-exploration"):
    for i, prompt in enumerate(prompts):
        result = query_ollama(prompt)
        print(f"\n--- Prompt {i+1} ---\n{prompt}\n--- Response ---\n{result}\n")
