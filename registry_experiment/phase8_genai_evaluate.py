import mlflow
import pandas as pd
import requests

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("llm-prompt-experiments")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:latest"

def query_ollama(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False
    })
    return response.json()["response"]

questions = [
    "Explain what a model registry is in one sentence.",
    "Explain what MLflow autologging does in one sentence.",
    "What is the capital of France?",
]
expected = [
    "A model registry is a centralized system for versioning, tracking, and managing ML models throughout their lifecycle.",
    "Autologging automatically captures parameters, metrics, and models during training without manual logging calls.",
    "Paris is the capital of France.",
]

eval_data = pd.DataFrame({
    "inputs": [{"query": q} for q in questions],          # dict per row now
    "outputs": [query_ollama(q) for q in questions],
    "expectations": [{"expected_response": e} for e in expected],  # renamed + dict
})

with mlflow.start_run(run_name="qwen2-genai-eval"):
    results = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[
            mlflow.genai.scorers.Correctness(model="ollama:/qwen2"),
            mlflow.genai.scorers.RelevanceToQuery(model="ollama:/qwen2"),
        ],
    )
    print("Eval results:", results.metrics)
