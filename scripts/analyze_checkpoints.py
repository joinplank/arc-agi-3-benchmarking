#!/usr/bin/env python3
"""
Analyze checkpoints and generate a report on model test results.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

def analyze_checkpoint(checkpoint_dir: Path) -> Dict[str, Any]:
    """Analyze a single checkpoint directory."""
    result = {
        'exists': False,
        'has_metadata': False,
        'has_costs': False,
        'has_action_history': False,
        'has_memory': False,
        'has_previous_images': False,
        'config': None,
        'game_id': None,
        'action_counter': None,
        'play_action_counter': None,
        'errors': []
    }
    
    if not checkpoint_dir.exists():
        result['errors'].append(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return result
    
    result['exists'] = True
    
    # Check metadata.json
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        result['has_metadata'] = True
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                result['config'] = metadata.get('config')
                result['game_id'] = metadata.get('game_id')
                result['action_counter'] = metadata.get('action_counter', 0)
                result['play_action_counter'] = metadata.get('play_action_counter', 0)
        except Exception as e:
            result['errors'].append(f"Error reading metadata.json: {e}")
    else:
        result['errors'].append("Missing metadata.json")
    
    # Check costs.json
    costs_path = checkpoint_dir / "costs.json"
    if costs_path.exists():
        result['has_costs'] = True
        try:
            with open(costs_path) as f:
                costs = json.load(f)
                if not costs.get('total_cost') or not costs.get('total_usage'):
                    result['errors'].append("costs.json missing required fields")
        except Exception as e:
            result['errors'].append(f"Error reading costs.json: {e}")
    else:
        result['errors'].append("Missing costs.json")
    
    # Check action_history.json
    action_history_path = checkpoint_dir / "action_history.json"
    if action_history_path.exists():
        result['has_action_history'] = True
        try:
            with open(action_history_path) as f:
                action_history = json.load(f)
                if not isinstance(action_history, list):
                    result['errors'].append("action_history.json is not a list")
        except Exception as e:
            result['errors'].append(f"Error reading action_history.json: {e}")
    else:
        result['errors'].append("Missing action_history.json")
    
    # Check memory.txt
    memory_path = checkpoint_dir / "memory.txt"
    if memory_path.exists():
        result['has_memory'] = True
    
    # Check previous_images directory
    images_dir = checkpoint_dir / "previous_images"
    if images_dir.exists() and any(images_dir.iterdir()):
        result['has_previous_images'] = True
    
    return result

def parse_test_log(log_file: str) -> Dict[str, Any]:
    """Parse the test output log to extract model results."""
    # Resolve log file path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    log_path = os.path.join(project_root, log_file) if not os.path.isabs(log_file) else log_file
    
    results = {
        'successful': [],
        'failed': [],
        'missing_checkpoints': [],
        'errors': defaultdict(list)
    }
    
    if not os.path.exists(log_path):
        return results
    
    current_model = None
    with open(log_path, 'r') as f:
        for line in f:
            # Look for model start
            if "Starting test for model:" in line:
                current_model = line.split("Starting test for model:")[-1].strip()
            
            # Look for success
            if current_model and "✓" in line and "Checkpoint verified" in line:
                if current_model not in results['successful']:
                    results['successful'].append(current_model)
            
            # Look for errors
            if current_model and "✗" in line:
                error_msg = line.split("✗")[-1].strip()
                if current_model not in results['failed']:
                    results['failed'].append(current_model)
                results['errors'][current_model].append(error_msg)
            
            # Look for missing checkpoints
            if current_model and "No card_id returned" in line:
                if current_model not in results['missing_checkpoints']:
                    results['missing_checkpoints'].append(current_model)
    
    return results

def generate_report():
    """Generate comprehensive checkpoint analysis report."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    checkpoint_dir = project_root / ".checkpoint"
    log_file = "test_run_output.log"
    
    print("="*80)
    print("CHECKPOINT ANALYSIS REPORT".center(80))
    print("="*80)
    print()
    
    # Parse test log
    log_results = parse_test_log(log_file)
    
    # Analyze all checkpoints
    checkpoint_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    print(f"Found {len(checkpoint_dirs)} checkpoint directories")
    print()
    
    checkpoint_analyses = {}
    for cp_dir in checkpoint_dirs:
        analysis = analyze_checkpoint(cp_dir)
        checkpoint_analyses[cp_dir.name] = analysis
    
    # Group by model config
    models_by_config = defaultdict(list)
    for cp_id, analysis in checkpoint_analyses.items():
        if analysis['config']:
            models_by_config[analysis['config']].append({
                'checkpoint_id': cp_id,
                'analysis': analysis
            })
    
    # Report statistics
    print("="*80)
    print("CHECKPOINT STATISTICS".center(80))
    print("="*80)
    print()
    
    total_checkpoints = len(checkpoint_analyses)
    valid_checkpoints = sum(1 for a in checkpoint_analyses.values() 
                           if a['has_metadata'] and a['has_costs'] and a['has_action_history'])
    invalid_checkpoints = total_checkpoints - valid_checkpoints
    
    print(f"Total Checkpoints: {total_checkpoints}")
    print(f"Valid Checkpoints: {valid_checkpoints}")
    print(f"Invalid Checkpoints: {invalid_checkpoints}")
    print()
    
    # Checkpoint completeness
    print("Checkpoint Completeness:")
    print(f"  • Has metadata.json: {sum(1 for a in checkpoint_analyses.values() if a['has_metadata'])}")
    print(f"  • Has costs.json: {sum(1 for a in checkpoint_analyses.values() if a['has_costs'])}")
    print(f"  • Has action_history.json: {sum(1 for a in checkpoint_analyses.values() if a['has_action_history'])}")
    print(f"  • Has memory.txt: {sum(1 for a in checkpoint_analyses.values() if a['has_memory'])}")
    print(f"  • Has previous_images: {sum(1 for a in checkpoint_analyses.values() if a['has_previous_images'])}")
    print()
    
    # Models with checkpoints
    print("="*80)
    print("MODELS WITH CHECKPOINTS".center(80))
    print("="*80)
    print()
    
    models_with_checkpoints = set()
    for analysis in checkpoint_analyses.values():
        if analysis['config']:
            models_with_checkpoints.add(analysis['config'])
    
    print(f"Total unique models with checkpoints: {len(models_with_checkpoints)}")
    print()
    
    # Models by provider
    print("Models by Provider:")
    provider_counts = defaultdict(int)
    for analysis in checkpoint_analyses.values():
        if analysis['config']:
            # Try to infer provider from config name
            config = analysis['config']
            if 'claude' in config.lower() or 'anthropic' in config.lower():
                provider_counts['Anthropic'] += 1
            elif 'gpt' in config.lower() or 'o1' in config.lower() or 'o3' in config.lower() or 'o4' in config.lower():
                provider_counts['OpenAI'] += 1
            elif 'gemini' in config.lower():
                provider_counts['Gemini'] += 1
            elif 'grok' in config.lower():
                provider_counts['Grok/X.AI'] += 1
            elif 'deepseek' in config.lower():
                provider_counts['DeepSeek'] += 1
            elif 'magistral' in config.lower() or 'mistral' in config.lower():
                provider_counts['Mistral'] += 1
            elif 'qwen' in config.lower():
                provider_counts['Qwen'] += 1
            else:
                provider_counts['Other'] += 1
    
    for provider, count in sorted(provider_counts.items()):
        print(f"  • {provider}: {count}")
    print()
    
    # Checkpoints with errors
    print("="*80)
    print("CHECKPOINTS WITH ERRORS".center(80))
    print("="*80)
    print()
    
    checkpoints_with_errors = {cp_id: analysis for cp_id, analysis in checkpoint_analyses.items() 
                               if analysis['errors']}
    
    if checkpoints_with_errors:
        print(f"Found {len(checkpoints_with_errors)} checkpoints with errors:")
        print()
        for cp_id, analysis in list(checkpoints_with_errors.items())[:20]:  # Show first 20
            print(f"  • {cp_id}")
            print(f"    Config: {analysis['config'] or 'Unknown'}")
            for error in analysis['errors']:
                print(f"    - {error}")
        if len(checkpoints_with_errors) > 20:
            print(f"    ... and {len(checkpoints_with_errors) - 20} more")
    else:
        print("No checkpoints with errors found!")
    print()
    
    # Models that failed
    print("="*80)
    print("MODELS THAT FAILED".center(80))
    print("="*80)
    print()
    
    if log_results['failed']:
        print(f"Found {len(log_results['failed'])} models that failed:")
        print()
        for model in sorted(log_results['failed'])[:30]:  # Show first 30
            errors = log_results['errors'].get(model, [])
            print(f"  • {model}")
            if errors:
                for error in errors[:2]:  # Show first 2 errors
                    print(f"    - {error}")
        if len(log_results['failed']) > 30:
            print(f"    ... and {len(log_results['failed']) - 30} more")
    else:
        print("No failed models found in log!")
    print()
    
    # Models missing checkpoints
    print("="*80)
    print("MODELS MISSING CHECKPOINTS".center(80))
    print("="*80)
    print()
    
    if log_results['missing_checkpoints']:
        print(f"Found {len(log_results['missing_checkpoints'])} models missing checkpoints:")
        for model in sorted(log_results['missing_checkpoints'])[:20]:
            print(f"  • {model}")
        if len(log_results['missing_checkpoints']) > 20:
            print(f"    ... and {len(log_results['missing_checkpoints']) - 20} more")
    else:
        print("No models missing checkpoints!")
    print()
    
    # Successful models
    print("="*80)
    print("SUCCESSFUL MODELS".center(80))
    print("="*80)
    print()
    
    if log_results['successful']:
        print(f"Found {len(log_results['successful'])} models that completed successfully:")
        for model in sorted(log_results['successful'])[:30]:
            print(f"  • {model}")
        if len(log_results['successful']) > 30:
            print(f"    ... and {len(log_results['successful']) - 30} more")
    else:
        print("No successful models found in log (check log file)")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY".center(80))
    print("="*80)
    print()
    print(f"Total Models Tested: {len(log_results['successful']) + len(log_results['failed'])}")
    print(f"Successful: {len(log_results['successful'])}")
    print(f"Failed: {len(log_results['failed'])}")
    print(f"Models with Checkpoints: {len(models_with_checkpoints)}")
    print(f"Valid Checkpoints: {valid_checkpoints}")
    print(f"Invalid Checkpoints: {invalid_checkpoints}")
    print()
    print("="*80)

if __name__ == "__main__":
    generate_report()

