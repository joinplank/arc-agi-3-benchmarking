import os
from arcagi3.schemas import ModelConfig, GameResult
from typing import Optional
import json
import yaml


def read_models_config(config: str) -> ModelConfig:
    """
    Reads and parses models.yml configuration file for a specific configuration.
    
    Args:
        config: The configuration name to look up (e.g., 'gpt-4o-2024-11-20')
        
    Returns:
        ModelConfig: The configuration for the specified model
        
    Raises:
        ValueError: If no matching configuration is found
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(base_dir, "models.yml")
    models_private_file = os.path.join(base_dir, "models_private.yml")
    
    # Initialize with models from the main config file
    with open(models_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Add models from private config if it exists
    if os.path.exists(models_private_file):
        with open(models_private_file, 'r') as f:
            private_config_data = yaml.safe_load(f)
            # Merge the models lists
            if 'models' in private_config_data:
                config_data['models'].extend(private_config_data['models'])
    
    # Look for a model with the name matching the config parameter
    for model in config_data['models']:
        if model.get('name') == config:
            return ModelConfig(**model)
            
    raise ValueError(f"No matching configuration found for '{config}'")


def result_exists(save_results_dir: str, game_id: str) -> bool:
    """
    Check if a result file already exists for a given game.
    
    Args:
        save_results_dir: Directory where results are saved
        game_id: The game ID to check
        
    Returns:
        bool: True if result file exists, False otherwise
    """
    if not save_results_dir:
        return False
    
    # Check for any file that starts with the game_id
    if not os.path.exists(save_results_dir):
        return False
    
    for filename in os.listdir(save_results_dir):
        if filename.startswith(f"{game_id}_") and filename.endswith('.json'):
            return True
    
    return False


def save_result(save_results_dir: str, game_result: GameResult) -> str:
    """
    Save the game result to a JSON file.
    
    Args:
        save_results_dir: Directory to save results
        game_result: GameResult object to save
        
    Returns:
        str: Path to the saved file
    """
    os.makedirs(save_results_dir, exist_ok=True)
    
    # Create filename with game_id, config, and timestamp
    timestamp_str = game_result.timestamp.strftime("%Y%m%d_%H%M%S") if game_result.timestamp else "unknown"
    result_file = os.path.join(
        save_results_dir,
        f"{game_result.game_id}_{game_result.config}_{timestamp_str}.json"
    )
    
    with open(result_file, "w") as f:
        json.dump(game_result.model_dump(mode='json'), f, indent=2, default=str)
    
    return result_file

