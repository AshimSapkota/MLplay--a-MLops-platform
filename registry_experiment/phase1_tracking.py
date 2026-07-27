import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Point at your local server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Experiments group related runs (e.g. "fraud-detection-model")
mlflow.set_experiment("customer-churn-prediction")

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try 3 different hyperparameter configs — simulating real experimentation
configs = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]

for i, params in enumerate(configs):
    with mlflow.start_run(run_name=f"rf-config-{i+1}"):
        # Log parameters — what inputs produced this model
        mlflow.log_params(params)
        mlflow.log_param("dataset_version", "v1_synthetic")

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Log metrics — how well it performed
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log the model artifact itself
        mlflow.sklearn.log_model(model, "model")

        print(f"Run {i+1}: acc={acc:.3f}, f1={f1:.3f}")
