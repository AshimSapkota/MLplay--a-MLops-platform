"""
register_ollama_llm.py

Registers an Ollama-served (fine-tuned) LLM into the MLflow Model Registry as
a pyfunc wrapper. The actual weights stay in Ollama's own model store — this
registry entry is a versioned, aliasable POINTER with a real signature and
governance tags, not a copy of the model file itself.

Why do this instead of just calling Ollama directly from the dashboard?
- Versioning: "qwen2-churn-finetune-v3" vs "v4" becomes a real registry version,
  not a string in someone's notes.
- Aliases: @champion / @challenger works exactly like the sklearn model.
- Governance: same required tags (intended_use, known_limitations, approved_by)
  and the same CI gate script can check this model too.
- Uniform serving: `mlflow models serve` / dashboard code can treat this
  identically to any other registered model, without special-casing "the LLM."
"""

import mlflow
import requests
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel


class OllamaLLMWrapper(PythonModel):
    """
    A thin adapter: MLflow's pyfunc interface in, Ollama's REST API out.
    This is what gets pickled/logged — NOT the model weights themselves.
    """

    def __init__(self, ollama_model_name: str, ollama_url: str = "http://localhost:11434/api/generate"):
        self.ollama_model_name = ollama_model_name
        self.ollama_url = ollama_url

    def predict(self, context, model_input: pd.DataFrame, params=None):
        # Expects a DataFrame with a single "prompt" column — one row per request
        prompts = model_input["prompt"].tolist()
        responses = []
        for prompt in prompts:
            resp = requests.post(self.ollama_url, json={
                "model": self.ollama_model_name,
                "prompt": prompt,
                "stream": False,
            }, timeout=120)
            resp.raise_for_status()
            responses.append(resp.json().get("response", ""))
        return responses


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("llm-prompt-experiments")

    MODEL_NAME = "support-churn-explainer-llm"        # matches naming convention
    OLLAMA_MODEL_TAG = "qwen2:latest"          # <-- the actual name of YOUR finetuned
                                                         #     model as it exists in `ollama list`

    wrapper = OllamaLLMWrapper(ollama_model_name=OLLAMA_MODEL_TAG)

    # Build a real signature from an actual example call — same discipline as the sklearn model
    example_input = pd.DataFrame({"prompt": ["Explain what a model registry is in one sentence."]})
    example_output = wrapper.predict(None, example_input)
    signature = infer_signature(example_input, example_output)

    with mlflow.start_run(run_name="register-ollama-finetuned-llm"):
        model_info = mlflow.pyfunc.log_model(
            python_model=wrapper,
            name="model",
            signature=signature,
            input_example=example_input,
            registered_model_name=MODEL_NAME,
        )
        version = model_info.registered_model_version

        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

        # Same governance tags as the classic ML model — the pointer is
        # honestly documented, same as any other registry entry.
        client.set_model_version_tag(MODEL_NAME, version, "intended_use",
            "Explain churn-risk factors in plain language for internal support agents. Not for customer-facing automated responses.")
        client.set_model_version_tag(MODEL_NAME, version, "known_limitations",
            "Finetuned on limited internal data; may hallucinate on questions outside the churn-support domain.")
        client.set_model_version_tag(MODEL_NAME, version, "backing_runtime", "ollama")
        client.set_model_version_tag(MODEL_NAME, version, "ollama_model_name", OLLAMA_MODEL_TAG)

        print(f"Registered {MODEL_NAME} version {version} (Ollama-backed pointer).")
