import numpy as np
import pandas as pd
import pickle
import os
import json
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer."""
    try:
        root_dir = get_root_directory()
        
        # Load the model
        model_path = os.path.join(root_dir, 'lgbm_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully")
        
        # Load the vectorizer
        vectorizer_path = os.path.join(root_dir, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.debug("Vectorizer loaded successfully")
        
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error in loading model or vectorizer: {e}")
        raise

def load_test_data():
    """Load the test data."""
    try:
        root_dir = get_root_directory()
        test_path = os.path.join(root_dir, 'data/interim/test_processed.csv')
        test_data = pd.read_csv(test_path)
        logger.debug("Test data loaded successfully")
        return test_data
    except Exception as e:
        logger.error(f"Error in loading test data: {e}")
        raise

def evaluate_model(model, vectorizer, test_data):
    """Evaluate the model and generate metrics."""
    try:
        # Transform test data using the vectorizer
        X_test = vectorizer.transform(test_data['clean_comment'].values)
        y_test = test_data['category'].values
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save metrics to JSON
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        root_dir = get_root_directory()
        with open(os.path.join(root_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(root_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        print("========================")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        logger.debug("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

def main():
    try:
        # Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer()
        
        # Load test data
        test_data = load_test_data()
        
        # Evaluate model
        evaluate_model(model, vectorizer, test_data)
        
    except Exception as e:
        logger.error(f"Error in model evaluation process: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()



