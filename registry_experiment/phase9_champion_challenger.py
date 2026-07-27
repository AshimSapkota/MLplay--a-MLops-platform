import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import evaluate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("customer-churn-prediction")
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

MODEL_NAME = "churn-predictor"

# Same eval set for both — apples to apples comparison
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_cols)
df["target"] = y
_, X_test, _, y_test = train_test_split(df[feature_cols], df["target"], test_size=0.2, random_state=42)
eval_df = X_test.copy()
eval_df["target"] = y_test

results = {}

for role, alias in [("champion", "champion"), ("challenger", "challenger")]:
    with mlflow.start_run(run_name=f"ab-eval-{role}") as run:
        mlflow.set_tag("comparison_role", role)  # tag, not just run name — queryable later
        version = client.get_model_version_by_alias(MODEL_NAME, alias)
        mlflow.set_tag("model_version", version.version)

        result = evaluate(
            model=f"models:/{MODEL_NAME}@{alias}",
            data=eval_df,
            targets="target",
            model_type="classifier",
        )
        results[role] = {
            "run_id": run.info.run_id,
            "version": version.version,
            "metrics": result.metrics,
        }
        print(f"{role} (v{version.version}): accuracy={result.metrics['accuracy_score']:.3f}, "
              f"f1={result.metrics['f1_score']:.3f}, recall={result.metrics['recall_score']:.3f}")

# --- Compare and log the decision itself ---
champ_f1 = results["champion"]["metrics"]["f1_score"]
chall_f1 = results["challenger"]["metrics"]["f1_score"]
winner = "challenger" if chall_f1 > champ_f1 else "champion"

with mlflow.start_run(run_name="ab-comparison-summary"):
    mlflow.log_metric("champion_f1", champ_f1)
    mlflow.log_metric("challenger_f1", chall_f1)
    mlflow.log_metric("f1_delta", chall_f1 - champ_f1)
    mlflow.set_tag("decision", winner)
    mlflow.set_tag("champion_version", results["champion"]["version"])
    mlflow.set_tag("challenger_version", results["challenger"]["version"])

print(f"\nDecision: {winner} wins (champion f1={champ_f1:.3f} vs challenger f1={chall_f1:.3f})")

# --- Promote only if challenger actually won ---
if winner == "challenger":
    print(f"Promoting version {results['challenger']['version']} to champion...")
    client.set_registered_model_alias(MODEL_NAME, "champion", results["challenger"]["version"])
else:
    print("Champion retained — no promotion.")
