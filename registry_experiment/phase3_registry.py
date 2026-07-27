import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("customer-churn-prediction")

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_cols)
df["target"] = y
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols],df["target"], test_size=0.2, random_state=42)

MODEL_NAME = "churn-predictor"  # this is the registry entity name

# --- Version 1: Random Forest ---
with mlflow.start_run(run_name="register-v1-rf"):

    train_df = X_train.copy()
    train_df["target"] = y_train
    dataset = mlflow.data.from_pandas(
        train_df,
        source="synthetic-make_classification-v1",  # in real use: S3 path, DB table, DVC hash, etc.
        name="churn-training-data",
        targets="target"
    )

    # Log it against this run — this is the key call
    mlflow.log_input(dataset, context="training")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    # This is the key call — logs model AND registers it as a new version
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=MODEL_NAME
    )
    print(f"v1 (RF) accuracy: {acc:.3f}")

# --- Version 2: Gradient Boosting (better model, new version) ---
with mlflow.start_run(run_name="register-v2-gbc"):

    train_df = X_train.copy()
    train_df["target"] = y_train
    dataset = mlflow.data.from_pandas(
        train_df,
        source="synthetic-make_classification-v1",  # in real use: S3 path, DB table, DVC hash, etc.
        name="churn-training-data",
        targets="target"
    )

    # Log it against this run — this is the key call
    mlflow.log_input(dataset, context="training")
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=MODEL_NAME  # same name → creates version 2
    )
    print(f"v2 (GBC) accuracy: {acc:.3f}")
