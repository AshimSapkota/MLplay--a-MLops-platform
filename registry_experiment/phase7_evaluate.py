import mlflow
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("customer-churn-prediction")

# Recreate the same-shaped eval data (matching training schema)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_cols)
df["target"] = y

_, X_test, _, y_test = train_test_split(df[feature_cols], df["target"], test_size=0.2, random_state=42)
eval_df = X_test.copy()
eval_df["target"] = y_test

with mlflow.start_run(run_name="evaluate-champion"):
    result = mlflow.evaluate(
        model="models:/churn-predictor@champion",  # loads by alias, like before
        data=eval_df,
        targets="target",
        model_type="classifier",
    )
    print("Metrics:", result.metrics)
