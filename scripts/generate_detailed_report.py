#!/usr/bin/env python3
"""
Generate detailed report on model test results and checkpoint validation.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import subprocess

def get_all_models():
    """Get all model names from models.yml"""
    import yaml
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_file = project_root / "src/arcagi3/models.yml"
    with open(models_file, 'r') as f:
        config_data = yaml.safe_load(f)
    return [model['name'] for model in config_data.get('models', [])]

def analyze_all_checkpoints():
    """Analyze all checkpoints and return detailed information."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    checkpoint_dir = project_root / ".checkpoint"
    results = {}
    
    for cp_dir in checkpoint_dir.iterdir():
        if not cp_dir.is_dir():
            continue
        
        cp_id = cp_dir.name
        metadata_path = cp_dir / "metadata.json"
        
        if not metadata_path.exists():
            continue
        
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            config = metadata.get('config')
            game_id = metadata.get('game_id')
            action_counter = metadata.get('action_counter', 0)
            play_action_counter = metadata.get('play_action_counter', 0)
            
            # Check other files
            has_costs = (cp_dir / "costs.json").exists()
            has_action_history = (cp_dir / "action_history.json").exists()
            has_memory = (cp_dir / "memory.txt").exists()
            
            # Count actions in history
            action_count = 0
            if has_action_history:
                try:
                    with open(cp_dir / "action_history.json") as f:
                        action_history = json.load(f)
                        action_count = len(action_history) if isinstance(action_history, list) else 0
                except:
                    pass
            
            if config not in results:
                results[config] = {
                    'checkpoints': [],
                    'total_actions': 0,
                    'max_actions': 0,
                    'min_actions': float('inf'),
                    'has_all_files': True
                }
            
            checkpoint_info = {
                'checkpoint_id': cp_id,
                'game_id': game_id,
                'action_counter': action_counter,
                'play_action_counter': play_action_counter,
                'action_count': action_count,
                'has_costs': has_costs,
                'has_action_history': has_action_history,
                'has_memory': has_memory
            }
            
            results[config]['checkpoints'].append(checkpoint_info)
            results[config]['total_actions'] += action_count
            results[config]['max_actions'] = max(results[config]['max_actions'], action_count)
            if action_count > 0:
                results[config]['min_actions'] = min(results[config]['min_actions'], action_count)
            
            if not (has_costs and has_action_history and has_memory):
                results[config]['has_all_files'] = False
                
        except Exception as e:
            print(f"Error analyzing {cp_dir}: {e}")
    
    return results

def generate_report():
    """Generate comprehensive report."""
    print("="*80)
    print("DETAILED MODEL TEST REPORT".center(80))
    print("="*80)
    print()
    
    # Get all models
    all_models = get_all_models()
    print(f"Total models in config: {len(all_models)}")
    print()
    
    # Analyze checkpoints
    checkpoint_results = analyze_all_checkpoints()
    
    print("="*80)
    print("CHECKPOINT SUMMARY".center(80))
    print("="*80)
    print()
    
    models_with_checkpoints = set(checkpoint_results.keys())
    models_without_checkpoints = set(all_models) - models_with_checkpoints
    
    print(f"Models with checkpoints: {len(models_with_checkpoints)}")
    print(f"Models without checkpoints: {len(models_without_checkpoints)}")
    print()
    
    # Checkpoint validation
    print("="*80)
    print("CHECKPOINT VALIDATION".center(80))
    print("="*80)
    print()
    
    valid_models = []
    invalid_models = []
    
    for model, data in checkpoint_results.items():
        if data['has_all_files'] and len(data['checkpoints']) > 0:
            valid_models.append(model)
        else:
            invalid_models.append((model, data))
    
    print(f"Models with valid checkpoints: {len(valid_models)}")
    print(f"Models with invalid checkpoints: {len(invalid_models)}")
    print()
    
    if invalid_models:
        print("Invalid checkpoints:")
        for model, data in invalid_models[:10]:
            print(f"  • {model}")
            if not data['has_all_files']:
                print("    - Missing required files")
            if len(data['checkpoints']) == 0:
                print("    - No checkpoints found")
        if len(invalid_models) > 10:
            print(f"    ... and {len(invalid_models) - 10} more")
        print()
    
    # Action counts
    print("="*80)
    print("ACTION COUNTS BY MODEL".center(80))
    print("="*80)
    print()
    
    # Group by action count
    zero_actions = []
    one_action = []
    two_actions = []
    more_actions = []
    
    for model, data in checkpoint_results.items():
        max_actions = data['max_actions']
        if max_actions == 0:
            zero_actions.append(model)
        elif max_actions == 1:
            one_action.append(model)
        elif max_actions == 2:
            two_actions.append(model)
        else:
            more_actions.append((model, max_actions))
    
    print(f"Models with 0 actions: {len(zero_actions)}")
    if zero_actions:
        print("  Examples:", ", ".join(zero_actions[:5]))
        if len(zero_actions) > 5:
            print(f"    ... and {len(zero_actions) - 5} more")
    print()
    
    print(f"Models with 1 action: {len(one_action)}")
    if one_action:
        print("  Examples:", ", ".join(one_action[:5]))
        if len(one_action) > 5:
            print(f"    ... and {len(one_action) - 5} more")
    print()
    
    print(f"Models with 2 actions: {len(two_actions)}")
    print(f"  (Expected for max_actions=2)")
    print()
    
    print(f"Models with >2 actions: {len(more_actions)}")
    if more_actions:
        print("  (May indicate multiple runs or retries)")
        for model, count in more_actions[:10]:
            print(f"    • {model}: {count} actions")
        if len(more_actions) > 10:
            print(f"    ... and {len(more_actions) - 10} more")
    print()
    
    # Models by provider
    print("="*80)
    print("MODELS BY PROVIDER".center(80))
    print("="*80)
    print()
    
    provider_groups = defaultdict(list)
    for model in models_with_checkpoints:
        if 'claude' in model.lower() or 'anthropic' in model.lower():
            provider_groups['Anthropic'].append(model)
        elif 'gpt' in model.lower() or 'o1' in model.lower() or 'o3' in model.lower() or 'o4' in model.lower():
            provider_groups['OpenAI'].append(model)
        elif 'gemini' in model.lower():
            provider_groups['Gemini'].append(model)
        elif 'grok' in model.lower():
            provider_groups['Grok/X.AI'].append(model)
        elif 'deepseek' in model.lower():
            provider_groups['DeepSeek'].append(model)
        elif 'magistral' in model.lower() or 'mistral' in model.lower():
            provider_groups['Mistral'].append(model)
        elif 'qwen' in model.lower():
            provider_groups['Qwen'].append(model)
        else:
            provider_groups['Other'].append(model)
    
    for provider in sorted(provider_groups.keys()):
        models = provider_groups[provider]
        print(f"{provider}: {len(models)} models")
        print(f"  Models: {', '.join(sorted(models)[:10])}")
        if len(models) > 10:
            print(f"    ... and {len(models) - 10} more")
        print()
    
    # Models without checkpoints
    print("="*80)
    print("MODELS WITHOUT CHECKPOINTS".center(80))
    print("="*80)
    print()
    
    if models_without_checkpoints:
        print(f"Found {len(models_without_checkpoints)} models without checkpoints:")
        for model in sorted(models_without_checkpoints)[:20]:
            print(f"  • {model}")
        if len(models_without_checkpoints) > 20:
            print(f"    ... and {len(models_without_checkpoints) - 20} more")
    else:
        print("All models have checkpoints!")
    print()
    
    # Sample checkpoint validation
    print("="*80)
    print("SAMPLE CHECKPOINT VALIDATION".center(80))
    print("="*80)
    print()
    
    sample_models = list(checkpoint_results.keys())[:5]
    for model in sample_models:
        data = checkpoint_results[model]
        cp = data['checkpoints'][0]
        print(f"Model: {model}")
        print(f"  Checkpoint ID: {cp['checkpoint_id']}")
        print(f"  Game ID: {cp['game_id']}")
        print(f"  Actions: {cp['action_count']}")
        print(f"  Has costs.json: {cp['has_costs']}")
        print(f"  Has action_history.json: {cp['has_action_history']}")
        print(f"  Has memory.txt: {cp['has_memory']}")
        print()
    
    # Summary
    print("="*80)
    print("FINAL SUMMARY".center(80))
    print("="*80)
    print()
    print(f"Total models in config: {len(all_models)}")
    print(f"Models with checkpoints: {len(models_with_checkpoints)}")
    print(f"Models without checkpoints: {len(models_without_checkpoints)}")
    print(f"Models with valid checkpoints: {len(valid_models)}")
    print(f"Models with invalid checkpoints: {len(invalid_models)}")
    print(f"Models with 2 actions (expected): {len(two_actions)}")
    print(f"Models with 0 actions (potential issues): {len(zero_actions)}")
    print()
    print("="*80)

if __name__ == "__main__":
    generate_report()

