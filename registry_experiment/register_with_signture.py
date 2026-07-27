import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models import evaluate
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("customer-churn-prediction")
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

MODEL_NAME = "credit-churn-gbc"

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_cols)
df["target"] = y
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], df["target"], test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="register-with-signature"):
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)
    signature = infer_signature(X_train, predictions)

    model_info = mlflow.sklearn.log_model(
        model, name="model", signature=signature,
        input_example=X_train.iloc[:2], registered_model_name=MODEL_NAME,
    )

    client.set_model_version_tag(MODEL_NAME, model_info.registered_model_version,
        "intended_use", "Predict customer churn risk to prioritize retention outreach.")
    client.set_model_version_tag(MODEL_NAME, model_info.registered_model_version,
        "known_limitations", "Trained on synthetic data.")

    # NEW — this is what was missing
    eval_df = X_test.copy()
    eval_df["target"] = y_test
    evaluate(
        model=f"runs:/{mlflow.active_run().info.run_id}/model",
        data=eval_df, targets="target", model_type="classifier",
    )

    print(f"Registered version {model_info.registered_model_version} with signature + metrics + tags.")
