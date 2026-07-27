from mlflow import MlflowClient
import mlflow

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "churn-predictor"

# --- Assign aliases ---
# v4 (GBC, 0.905 acc) is our best model -> promote to "champion" (production)
client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version="4")

# v3 (RF, 0.885 acc) -> mark as "challenger" (candidate being evaluated against champion)
client.set_registered_model_alias(name=MODEL_NAME, alias="challenger", version="3")

print("Aliases set.")

# --- Verify ---
champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
print(f"Champion is version {champion_version.version}")

# --- Load the model by alias (NOT by version number) ---
# This is the key production pattern — serving code never hardcodes a version
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@champion")

# --- Run inference ---
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5, n_features=20, random_state=99)  # simulate new incoming data
predictions = model.predict(X)
print("Predictions from champion model:", predictions)
