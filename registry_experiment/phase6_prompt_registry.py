import mlflow
import requests

mlflow.set_tracking_uri("http://127.0.0.1:5000")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:latest"

# --- Register Prompt Version 1 ---
prompt_v1 = mlflow.genai.register_prompt(
    name="churn-explainer-prompt",
    template="Explain what {{ concept }} is in one sentence.",
    commit_message="Initial simple version",
)
print(f"Registered prompt version: {prompt_v1.version}")

# --- Register Prompt Version 2 — improved/refined ---
prompt_v2 = mlflow.genai.register_prompt(
    name="churn-explainer-prompt",
    template=(
        "You are an expert MLOps engineer. Explain what {{ concept }} is "
        "in one clear sentence, suitable for a junior data scientist."
    ),
    commit_message="Added persona and audience targeting",
)
print(f"Registered prompt version: {prompt_v2.version}")

# --- Load a specific prompt version and use it ---
loaded_prompt = mlflow.genai.load_prompt("prompts:/churn-explainer-prompt/2")
formatted = loaded_prompt.format(concept="a feature store")
print(f"\nFormatted prompt (v2): {formatted}")

# --- Use it in an actual traced LLM call, logging which prompt version was used ---
mlflow.set_experiment("llm-prompt-experiments")

@mlflow.trace(name="qwen2-with-registered-prompt")
def query_ollama(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    })
    return response.json()["response"]

with mlflow.start_run(run_name="run-with-prompt-v2"):
    mlflow.log_param("prompt_name", "churn-explainer-prompt")
    mlflow.log_param("prompt_version", loaded_prompt.version)

    result = query_ollama(formatted)
    print(f"\n--- Response ---\n{result}")
