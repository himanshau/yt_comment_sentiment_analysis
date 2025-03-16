import mlflow
import random
import dagshub

# Initialize DAGsHub MLflow tracking
dagshub.init(repo_owner='himanshau', repo_name='yt_comment_sentiment_analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/himanshau/yt_comment_sentiment_analysis.mlflow")

# Start MLflow run and log parameters & metrics
with mlflow.start_run():
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(1, 10))

print("Logged metrics successfully.")
