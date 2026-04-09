# register model

import json
import mlflow
import logging
import os

# Set up MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/rohitbedse/yt-comment-sentiment-analysis.mlflow")


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Transition the latest model version to Staging."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get all versions and find the latest
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise Exception(f"No versions found for model '{model_name}'")
        
        latest_version = max(versions, key=lambda v: int(v.version)).version
        logger.debug(f'Latest version of {model_name}: {latest_version}')
        
        # Transition the model to "Staging" stage
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Staging"
            )
        except Exception:
            # Fallback: use aliases if transition_model_version_stage is deprecated
            client.set_registered_model_alias(model_name, "staging", latest_version)
        
        logger.debug(f'Model {model_name} version {latest_version} transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model staging: %s', e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()