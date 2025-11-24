import yaml
import subprocess
import sys
import os

def get_openai_models():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        models_file = os.path.join(project_root, "src", "arcagi3", "models.yml")
        with open(models_file, "r") as f:
            data = yaml.safe_load(f)
        
        openai_models = []
        if "models" in data:
            for model in data["models"]:
                if model.get("provider") == "openai":
                    openai_models.append(model["name"])
        return openai_models
    except Exception as e:
        print(f"Error reading models.yml: {e}")
        return []

def run_test(model_name):
    print(f"Testing model: {model_name}...", end=" ", flush=True)
    
    # Command to run a minimal test: 1 action, specific game
    cmd = [
        "uv", "run", "python", "-m", "arcagi3.cli",
        "--games", "ft09-b8377d4b7815",
        "--config", model_name,
        "--max_actions", "1",
        "--num_plays", "1"
    ]
    
    try:
        # Run command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per model
        )
        
        if result.returncode == 0:
            print("PASS")
            return True
        else:
            print(f"FAIL (Exit Code: {result.returncode})")
            print(f"  Error Output: {result.stderr[:200]}...") # Print first 200 chars of error
            return False
            
    except subprocess.TimeoutExpired:
        print("FAIL (Timeout)")
        return False
    except Exception as e:
        print(f"FAIL (Exception: {e})")
        return False

def main():
    models = get_openai_models()
    if not models:
        print("No OpenAI models found.")
        return

    print(f"Found {len(models)} OpenAI models. Starting verification...")
    print("-" * 50)

    passed = []
    failed = []

    for model in models:
        if run_test(model):
            passed.append(model)
        else:
            failed.append(model)

    print("-" * 50)
    print(f"Verification Complete.")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed Models:")
        for m in failed:
            print(f"- {m}")

if __name__ == "__main__":
    main()
