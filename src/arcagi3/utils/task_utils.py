import os
from arcagi3.schemas import ModelConfig, GameResult
from typing import Optional, Dict, List, Any
import json
import yaml
from datetime import datetime


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


def save_result_in_timestamped_structure(
    timestamp_dir: str,
    game_result: GameResult
) -> str:
    """
    Save the game result in the new timestamped structure:
    {timestamp_dir}/{game_id}/{game_id}_{config}_{timestamp}.json
    
    Args:
        timestamp_dir: Base directory with timestamp (e.g., results/20251101_143022)
        game_result: GameResult object to save
        
    Returns:
        str: Path to the saved file
    """
    # Create game-specific directory
    game_dir = os.path.join(timestamp_dir, game_result.game_id)
    os.makedirs(game_dir, exist_ok=True)
    
    # Create filename with game_id, config, and timestamp
    timestamp_str = game_result.timestamp.strftime("%Y%m%d_%H%M%S") if game_result.timestamp else "unknown"
    result_file = os.path.join(
        game_dir,
        f"{game_result.game_id}_{game_result.config}_{timestamp_str}.json"
    )
    
    with open(result_file, "w") as f:
        json.dump(game_result.model_dump(mode='json'), f, indent=2, default=str)
    
    return result_file


def read_provider_rate_limits() -> dict:
    """
    Reads and parses the provider_config.yml file to get rate limit configurations.

    Assumes provider_config.yml is in the project root directory.

    Returns:
        dict: A dictionary where keys are provider names and values are dicts
              containing 'rate' and 'period'.
              Example: {'openai': {'rate': 60, 'period': 60}}

    Raises:
        FileNotFoundError: If provider_config.yml is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    # Get project root (assuming this is in src/arcagi3/utils/task_utils.py)
    current_file = os.path.abspath(__file__)
    # Go up: utils -> arcagi3 -> src -> project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    provider_config_file = os.path.join(base_dir, "provider_config.yml")

    if not os.path.exists(provider_config_file):
        raise FileNotFoundError(f"provider_config.yml not found at {provider_config_file}")

    with open(provider_config_file, 'r') as f:
        try:
            rate_limits_data = yaml.safe_load(f)
            if not isinstance(rate_limits_data, dict):
                raise yaml.YAMLError("provider_config.yml root should be a dictionary of providers.")
            # Basic validation for each provider's config
            for provider, limits in rate_limits_data.items():
                if not isinstance(limits, dict) or 'rate' not in limits or 'period' not in limits:
                    raise yaml.YAMLError(
                        f"Provider '{provider}' in provider_config.yml must have 'rate' and 'period' keys."
                    )
                if not isinstance(limits['rate'], int) or not isinstance(limits['period'], int):
                    raise yaml.YAMLError(
                        f"'rate' and 'period' for provider '{provider}' must be integers."
                    )
            return rate_limits_data
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing provider_config.yml: {e}")


def generate_execution_map(timestamp_dir: str) -> Dict[str, Any]:
    """
    Generate execution map JSON that maps all executions.
    
    Structure:
    {
        "execution_start": "2025-11-01T14:30:22Z",
        "games": {
            "ls20-fa137e247ce6": {
                "models": ["gpt-4o-2024-11-20", "claude_opus"],
                "result_files": [
                    "ls20-fa137e247ce6/ls20-fa137e247ce6_gpt-4o-2024-11-20_20251101_143022.json",
                    "ls20-fa137e247ce6/ls20-fa137e247ce6_claude_opus_20251101_143100.json"
                ]
            }
        }
    }
    
    Args:
        timestamp_dir: Base directory with timestamp
        
    Returns:
        dict: Execution map dictionary
    """
    execution_map = {
        "execution_start": None,
        "games": {}
    }
    
    # Extract timestamp from directory name (format: YYYYMMDD_HHMMSS)
    dir_name = os.path.basename(timestamp_dir)
    try:
        # Try to parse timestamp from directory name
        if len(dir_name) == 15 and dir_name.count('_') == 1:
            date_part, time_part = dir_name.split('_')
            if len(date_part) == 8 and len(time_part) == 6:
                dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                execution_map["execution_start"] = dt.isoformat() + "Z"
    except Exception:
        pass
    
    # Scan all game directories
    for item in os.listdir(timestamp_dir):
        item_path = os.path.join(timestamp_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # This is a game directory
            game_id = item
            result_files = []
            models = set()
            
            # Scan JSON files in game directory
            for filename in os.listdir(item_path):
                if filename.endswith('.json'):
                    result_files.append(f"{game_id}/{filename}")
                    # Extract model name from filename: {game_id}_{config}_{timestamp}.json
                    # Timestamp is always YYYYMMDD_HHMMSS (13 chars + underscore = 14 chars before .json)
                    # So we can split on the last underscore before the timestamp
                    if filename.startswith(f"{game_id}_") and filename.endswith('.json'):
                        # Remove game_id prefix and .json suffix
                        remaining = filename[len(f"{game_id}_"):-len('.json')]
                        # Find the last underscore that precedes a timestamp pattern (YYYYMMDD_HHMMSS)
                        # Timestamp is 15 chars: YYYYMMDD_HHMMSS
                        if len(remaining) > 15:
                            # Check if last 15 chars match timestamp pattern
                            potential_timestamp = remaining[-15:]
                            if '_' in potential_timestamp:
                                date_part, time_part = potential_timestamp.split('_')
                                if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
                                    # Found timestamp, extract model name
                                    model_name = remaining[:-16]  # Remove _YYYYMMDD_HHMMSS
                                    if model_name:
                                        models.add(model_name)
                        else:
                            models.add(remaining)
            
            if result_files:
                execution_map["games"][game_id] = {
                    "models": sorted(list(models)),
                    "result_files": sorted(result_files)
                }
    
    return execution_map


def generate_summary(timestamp_dir: str) -> Dict[str, Any]:
    """
    Generate summary JSON with aggregated statistics.
    
    Args:
        timestamp_dir: Base directory with timestamp
        
    Returns:
        dict: Summary dictionary with aggregated stats
    """
    summary = {
        "execution_start": None,
        "execution_end": None,
        "total_games": 0,
        "total_executions": 0,
        "models_tested": [],
        "games_by_model": {},
        "stats_by_model": {},
        "overall_stats": {
            "total_cost": 0.0,
            "total_tokens": 0,
            "total_duration_seconds": 0.0,
            "wins": 0,
            "game_overs": 0,
            "in_progress": 0,
            "avg_score": 0.0,
            "avg_actions": 0.0
        }
    }
    
    # Extract timestamp from directory name
    dir_name = os.path.basename(timestamp_dir)
    try:
        if len(dir_name) == 15 and dir_name.count('_') == 1:
            date_part, time_part = dir_name.split('_')
            if len(date_part) == 8 and len(time_part) == 6:
                dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                summary["execution_start"] = dt.isoformat() + "Z"
    except Exception:
        pass
    
    models_tested = set()
    all_results = []
    latest_timestamp = None
    
    # Scan all game directories and load results
    for item in os.listdir(timestamp_dir):
        item_path = os.path.join(timestamp_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            game_id = item
            
            # Load all JSON files for this game
            for filename in os.listdir(item_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(item_path, filename)
                    try:
                        with open(file_path, 'r') as f:
                            result_data = json.load(f)
                            all_results.append(result_data)
                            
                            config = result_data.get('config', 'unknown')
                            models_tested.add(config)
                            
                            # Track latest timestamp
                            timestamp_str = result_data.get('timestamp')
                            if timestamp_str:
                                try:
                                    result_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    if latest_timestamp is None or result_dt > latest_timestamp:
                                        latest_timestamp = result_dt
                                except Exception:
                                    pass
                            
                            # Track games by model
                            if config not in summary["games_by_model"]:
                                summary["games_by_model"][config] = []
                            summary["games_by_model"][config].append(game_id)
                    except Exception as e:
                        continue
    
    summary["models_tested"] = sorted(list(models_tested))
    summary["total_games"] = len(set(os.path.basename(item) for item in os.listdir(timestamp_dir) 
                                     if os.path.isdir(os.path.join(timestamp_dir, item)) and not item.startswith('.')))
    summary["total_executions"] = len(all_results)
    
    if latest_timestamp:
        summary["execution_end"] = latest_timestamp.isoformat() + "Z"
    
    # Calculate stats by model
    for model in models_tested:
        model_results = [r for r in all_results if r.get('config') == model]
        
        total_cost = sum(r.get('total_cost', {}).get('total_cost', 0.0) for r in model_results)
        total_tokens = sum(r.get('usage', {}).get('total_tokens', 0) for r in model_results)
        total_duration = sum(r.get('duration_seconds', 0.0) for r in model_results)
        wins = sum(1 for r in model_results if r.get('final_state') == 'WIN')
        game_overs = sum(1 for r in model_results if r.get('final_state') == 'GAME_OVER')
        in_progress = sum(1 for r in model_results if r.get('final_state') == 'IN_PROGRESS')
        scores = [r.get('final_score', 0) for r in model_results]
        actions = [r.get('actions_taken', 0) for r in model_results]
        
        summary["stats_by_model"][model] = {
            "total_games": len(model_results),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_duration_seconds": total_duration,
            "avg_cost_per_game": total_cost / len(model_results) if model_results else 0.0,
            "avg_tokens_per_game": total_tokens / len(model_results) if model_results else 0.0,
            "avg_duration_per_game": total_duration / len(model_results) if model_results else 0.0,
            "wins": wins,
            "game_overs": game_overs,
            "in_progress": in_progress,
            "win_rate": wins / len(model_results) if model_results else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_actions": sum(actions) / len(actions) if actions else 0.0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0
        }
    
    # Calculate overall stats
    if all_results:
        summary["overall_stats"]["total_cost"] = sum(r.get('total_cost', {}).get('total_cost', 0.0) for r in all_results)
        summary["overall_stats"]["total_tokens"] = sum(r.get('usage', {}).get('total_tokens', 0) for r in all_results)
        summary["overall_stats"]["total_duration_seconds"] = sum(r.get('duration_seconds', 0.0) for r in all_results)
        summary["overall_stats"]["wins"] = sum(1 for r in all_results if r.get('final_state') == 'WIN')
        summary["overall_stats"]["game_overs"] = sum(1 for r in all_results if r.get('final_state') == 'GAME_OVER')
        summary["overall_stats"]["in_progress"] = sum(1 for r in all_results if r.get('final_state') == 'IN_PROGRESS')
        
        scores = [r.get('final_score', 0) for r in all_results]
        actions = [r.get('actions_taken', 0) for r in all_results]
        
        summary["overall_stats"]["avg_score"] = sum(scores) / len(scores) if scores else 0.0
        summary["overall_stats"]["avg_actions"] = sum(actions) / len(actions) if actions else 0.0
    
    return summary

