from mlflow import MlflowClient
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
client.set_registered_model_alias("aws_testrun", "champion", "1")
