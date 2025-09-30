import yaml
from typing import Dict, Any
import os
import openai
from pathlib import Path

def load_config_from_yaml(file_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and return as dictionary.
    
    Args:
        file_path: Path to config file. If None, auto-detects '../configs/config.yaml'
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Usage:
        # Auto-detect config file
        config = load_config_from_yaml()
        
        # Or specify custom path
        config = load_config_from_yaml('../configs/config.yaml')
    """
    # If no file path provided, using default location
    if file_path is None:
        # Getting the path relative to this file's location
        current_file = Path(__file__).resolve()  
        utils_dir = current_file.parent          
        project_root = utils_dir.parent          
        default_config_path = project_root / 'configs' / 'config.yaml'
        file_path = str(default_config_path)
    
    # Converting to absolute path for better error handling
    abs_path = Path(file_path).resolve()
    
    # Check if file exists
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {abs_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Looking for: {abs_path}"
        )
    
    try:
        with open(abs_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        print(f"Configuration loaded from: {abs_path}")
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {abs_path}: {e}")
    except Exception as e:
        raise Exception(f"Error loading configuration from {abs_path}: {e}")

def get_default_config_path() -> str:
    """
    Get the default configuration file path.
    
    Returns:
        str: Absolute path to the default config file
    """
    current_file = Path(__file__).resolve()
    utils_dir = current_file.parent
    project_root = utils_dir.parent
    return str(project_root / 'configs' / 'config.yaml')

def load_openai_cfg(self):
    """
    Load OpenAI configuration settings.

    This function sets the OpenAI API configuration settings, including the API type, base URL,
    version, and API key. It is intended to be called at the beginning of the script or application
    to configure OpenAI settings.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    print("OpenAI configuration loaded successfully!")

       