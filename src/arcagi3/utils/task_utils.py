import os
from arcagi3.schemas import ModelConfig, GameResult
from typing import Optional, Dict, List, Any
import json
import yaml
from datetime import datetime

def read_models_config(config: str) -> ModelConfig:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(base_dir, "models.yml")
    models_private_file = os.path.join(base_dir, "models_private.yml")
    
    with open(models_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if os.path.exists(models_private_file):
        with open(models_private_file, 'r') as f:
            private_config_data = yaml.safe_load(f)
            if 'models' in private_config_data:
                config_data['models'].extend(private_config_data['models'])
    
    for model in config_data['models']:
        if model.get('name') == config:
            return ModelConfig(**model)
            
    raise ValueError(f"No matching configuration found for '{config}'")


def result_exists(save_results_dir: str, game_id: str) -> bool:
    if not save_results_dir or not os.path.exists(save_results_dir):
        return False
    return any(filename.startswith(f"{game_id}_") and filename.endswith('.json') 
               for filename in os.listdir(save_results_dir))


def save_result(save_results_dir: str, game_result: GameResult) -> str:
    os.makedirs(save_results_dir, exist_ok=True)
    timestamp_str = game_result.timestamp.strftime("%Y%m%d_%H%M%S") if game_result.timestamp else "unknown"
    result_file = os.path.join(
        save_results_dir,
        f"{game_result.game_id}_{game_result.config}_{timestamp_str}.json"
    )
    with open(result_file, "w") as f:
        json.dump(game_result.model_dump(mode='json'), f, indent=2, default=str)
    return result_file


def save_result_in_timestamped_structure(timestamp_dir: str, game_result: GameResult) -> str:
    game_dir = os.path.join(timestamp_dir, game_result.game_id)
    os.makedirs(game_dir, exist_ok=True)
    timestamp_str = game_result.timestamp.strftime("%Y%m%d_%H%M%S") if game_result.timestamp else "unknown"
    result_file = os.path.join(
        game_dir,
        f"{game_result.game_id}_{game_result.config}_{timestamp_str}.json"
    )
    with open(result_file, "w") as f:
        json.dump(game_result.model_dump(mode='json'), f, indent=2, default=str)
    return result_file


def read_provider_rate_limits() -> dict:
    current_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    provider_config_file = os.path.join(base_dir, "provider_config.yml")

    if not os.path.exists(provider_config_file):
        raise FileNotFoundError(f"provider_config.yml not found at {provider_config_file}")

    with open(provider_config_file, 'r') as f:
        rate_limits_data = yaml.safe_load(f)
        if not isinstance(rate_limits_data, dict):
            raise yaml.YAMLError("provider_config.yml root should be a dictionary of providers.")
        for provider, limits in rate_limits_data.items():
            if not isinstance(limits, dict) or 'rate' not in limits or 'period' not in limits:
                raise yaml.YAMLError(f"Provider '{provider}' must have 'rate' and 'period' keys.")
            if not isinstance(limits['rate'], int) or not isinstance(limits['period'], int):
                raise yaml.YAMLError(f"'rate' and 'period' for provider '{provider}' must be integers.")
        return rate_limits_data


def generate_execution_map(timestamp_dir: str) -> Dict[str, Any]:
    execution_map = {"execution_start": None, "games": {}}
    
    dir_name = os.path.basename(timestamp_dir)
    try:
        if len(dir_name) == 15 and dir_name.count('_') == 1:
            date_part, time_part = dir_name.split('_')
            if len(date_part) == 8 and len(time_part) == 6:
                dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                execution_map["execution_start"] = dt.isoformat() + "Z"
    except Exception:
        pass
    
    for item in os.listdir(timestamp_dir):
        item_path = os.path.join(timestamp_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            game_id = item
            result_files = []
            models = set()
            
            for filename in os.listdir(item_path):
                if filename.endswith('.json'):
                    result_files.append(f"{game_id}/{filename}")
                    if filename.startswith(f"{game_id}_") and filename.endswith('.json'):
                        remaining = filename[len(f"{game_id}_"):-len('.json')]
                        if len(remaining) > 15:
                            potential_timestamp = remaining[-15:]
                            if '_' in potential_timestamp:
                                date_part, time_part = potential_timestamp.split('_')
                                if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
                                    model_name = remaining[:-16]
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
    
    for item in os.listdir(timestamp_dir):
        item_path = os.path.join(timestamp_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            game_id = item
            for filename in os.listdir(item_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(item_path, filename)
                    try:
                        with open(file_path, 'r') as f:
                            result_data = json.load(f)
                            all_results.append(result_data)
                            config = result_data.get('config', 'unknown')
                            models_tested.add(config)
                            
                            timestamp_str = result_data.get('timestamp')
                            if timestamp_str:
                                try:
                                    result_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    if latest_timestamp is None or result_dt > latest_timestamp:
                                        latest_timestamp = result_dt
                                except Exception:
                                    pass
                            
                            if config not in summary["games_by_model"]:
                                summary["games_by_model"][config] = []
                            summary["games_by_model"][config].append(game_id)
                    except Exception:
                        continue
    
    summary["models_tested"] = sorted(list(models_tested))
    summary["total_games"] = len(set(os.path.basename(item) for item in os.listdir(timestamp_dir) 
                                     if os.path.isdir(os.path.join(timestamp_dir, item)) and not item.startswith('.')))
    summary["total_executions"] = len(all_results)
    
    if latest_timestamp:
        summary["execution_end"] = latest_timestamp.isoformat() + "Z"
    
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

