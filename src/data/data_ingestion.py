import pandas as pd
import numpy as np
import os
import sys
import logging
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded successfully")
        return params
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"YAML error: {e}")
        raise e 
    except Exception as e:
        logger.error(f"Error in loading parameters: {e}")
        raise e


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error in loading data: {e}")
        raise e


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        #remove rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug("preprosedding completed move to next")
        return df
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        raise e

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise e

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug("Data saved successfully in raw folder")
    except Exception as e:
        logger.error(f"Error in saving data: {e}")
        raise e



def main():
    try:
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        

        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))


    except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            

if __name__ == "__main__":
    main()

