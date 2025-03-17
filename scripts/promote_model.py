import mlflow
import argparse
import dagshub
from mlflow.tracking import MlflowClient

def promote_model(stage):
    """Promote model to specified stage"""
    # Initialize DAGsHub
    dagshub.init(repo_owner='himanshau', repo_name='yt_comment_sentiment_analysis', mlflow=True)
    
    # Set MLflow tracking URI to DAGsHub
    mlflow.set_tracking_uri("https://dagshub.com/himanshau/yt_comment_sentiment_analysis.mlflow")
    
    client = MlflowClient()
    
    # Get the latest model version
    model_versions = client.search_model_versions(f"name='lgbm_model'")
    latest_version = max(model_versions, key=lambda x: x.version)
    
    # Transition the model to the specified stage
    client.transition_model_version_stage(
        name="lgbm_model",
        version=latest_version.version,
        stage=stage
    )
    
    print(f"Model version {latest_version.version} promoted to {stage}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, help="Stage to promote model to (staging/production)")
    args = parser.parse_args()
    
    if args.stage not in ["staging", "production"]:
        raise ValueError("Stage must be either 'staging' or 'production'")
    
    promote_model(args.stage)

if __name__ == "__main__":
    main() 