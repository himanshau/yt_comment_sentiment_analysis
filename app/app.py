import mlflow
import dagshub
import os
from mlflow.tracking import MlflowClient

dagshub.init(repo_owner='himanshau', repo_name='yt_comment_sentiment_analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/himanshau/yt_comment_sentiment_analysis.mlflow")


def load_model(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

load_model_from_registry= load_model("lgbm_model", 1)
print("load_model_from_registry")
