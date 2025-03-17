# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import dagshub
import pickle
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    dagshub.init(repo_owner='himanshau', repo_name='yt_comment_sentiment_analysis', mlflow=True)    
    mlflow.set_tracking_uri("https://dagshub.com/himanshau/yt_comment_sentiment_analysis.mlflow")
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer

# Alternative model loading function
def load_local_model():
    try:
        # Try different model file paths
        model_paths = ["lgbm_model.pkl", "./lgbm_model.pkl", "../lgbm_model.pkl"]
        
        for path in model_paths:
            try:
                print(f"Trying to load model from {path}")
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                print(f"Successfully loaded model from {path}")
                return model
            except FileNotFoundError:
                print(f"Model file not found at {path}")
                continue
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
                continue
        
        raise FileNotFoundError("Could not find model file in any of the expected locations")
    except Exception as e:
        print(f"Error in load_local_model: {e}")
        raise

def load_local_vectorizer():
    try:
        # Try different vectorizer file paths
        vectorizer_paths = ["tfidf_vectorizer.pkl", "./tfidf_vectorizer.pkl", "../tfidf_vectorizer.pkl"]
        
        for path in vectorizer_paths:
            try:
                print(f"Trying to load vectorizer from {path}")
                vectorizer = joblib.load(path)
                print(f"Successfully loaded vectorizer from {path}")
                return vectorizer
            except FileNotFoundError:
                print(f"Vectorizer file not found at {path}")
                continue
            except Exception as e:
                print(f"Error loading vectorizer from {path}: {e}")
                continue
        
        raise FileNotFoundError("Could not find vectorizer file in any of the expected locations")
    except Exception as e:
        print(f"Error in load_local_vectorizer: {e}")
        raise

# Initialize the model and vectorizer
try:
    print("Attempting to load model and vectorizer...")
    # Try loading from MLflow
    try:
        print("Trying to load from MLflow...")
        model, vectorizer = load_model_and_vectorizer("lgbm_model", "1", "./tfidf_vectorizer.pkl")
        print("Successfully loaded from MLflow")
    except Exception as e:
        print(f"Error loading from MLflow: {e}")
        # Fallback to local files
        print("Falling back to local files...")
        model = load_local_model()
        vectorizer = load_local_vectorizer()
        print("Successfully loaded from local files")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model or vectorizer: {e}")
    # Create dummy model and vectorizer for testing
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.dummy import DummyClassifier
    print("Creating dummy model and vectorizer for testing")
    model = DummyClassifier(strategy="most_frequent").fit([[0]], [0])
    vectorizer = CountVectorizer().fit(["dummy text"])
    print("Dummy model and vectorizer created")

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert to DataFrame with feature names for MLflow model
        feature_names = vectorizer.get_feature_names_out()
        transformed_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )
        
        # Make predictions
        predictions = model.predict(transformed_df).tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in predict_with_timestamps: {e}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": f"Prediction failed: {str(e)}", "traceback": error_trace}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert to DataFrame with feature names for MLflow model
        feature_names = vectorizer.get_feature_names_out()
        transformed_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )
        
        # Make predictions
        predictions = model.predict(transformed_df).tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in predict: {e}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": f"Prediction failed: {str(e)}", "traceback": error_trace}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

@app.route('/test_model', methods=['GET'])
def test_model():
    try:
        # Test with a simple comment
        test_comment = "This is a great video, I really enjoyed it!"
        print(f"Test comment: {test_comment}")
        
        # Preprocess the comment
        preprocessed = preprocess_comment(test_comment)
        print(f"Preprocessed comment: {preprocessed}")
        
        # For MLflow PyFuncModel, we need to handle the transformation differently
        try:
            # First transform the text using the vectorizer
            transformed = vectorizer.transform([preprocessed])
            print(f"Transformed shape: {transformed.shape}")
            
            # Convert the sparse matrix to a dense array and create a DataFrame with the expected feature names
            feature_names = vectorizer.get_feature_names_out()
            transformed_df = pd.DataFrame(
                transformed.toarray(),
                columns=feature_names
            )
            
            # Make prediction using the transformed DataFrame
            prediction = model.predict(transformed_df)
            print(f"Prediction result: {prediction}")
            
        except Exception as pe:
            print(f"Error in prediction: {pe}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
        return jsonify({
            "status": "success",
            "test_comment": test_comment,
            "preprocessed": preprocessed,
            "prediction": str(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else str(prediction),
            "model_type": str(type(model)),
            "vectorizer_type": str(type(vectorizer))
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in test_model: {e}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": error_trace
        }), 500

@app.route('/check_model', methods=['GET'])
def check_model():
    try:
        model_info = {
            "model_loaded": model is not None,
            "model_type": str(type(model)),
            "vectorizer_loaded": vectorizer is not None,
            "vectorizer_type": str(type(vectorizer))
        }
        return jsonify(model_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/simple_test', methods=['GET'])
def simple_test():
    return jsonify({
        "status": "API is working",
        "time": str(pd.Timestamp.now())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)