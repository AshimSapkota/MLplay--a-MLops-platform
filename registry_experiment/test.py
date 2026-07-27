from mlflow import MlflowClient
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

# Try fetching version 1 explicitly
try:
    v1 = client.load_prompt("prompts:/churn-explainer-prompt/1")
    print("Version 1 exists:")
    print(v1)
except Exception as e:
    print(f"Version 1 NOT found: {e}")

# Try version 2 for comparison
try:
    v2 = client.load_prompt("prompts:/churn-explainer-prompt/2")
    print("\nVersion 2 exists:")
    print(v2)
except Exception as e:
    print(f"Version 2 NOT found: {e}")
