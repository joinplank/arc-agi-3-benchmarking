
import subprocess
import sys
import os

models = [
    "gemini-2-5-flash-preview-05-20",
    "gemini-2-5-flash-preview-05-20-thinking-1k",
    "gemini-2-5-flash-preview-05-20-thinking-8k",
    "gemini-2-5-flash-preview-05-20-thinking-16k",
    "gemini-2-5-flash-preview-05-20-thinking-24k",
    # "gemini-2-0-flash-001", # Already verified
    # "gemini-2-5-pro-preview-openrouter", # Already verified
    # "gemini-2-5-pro-preview-openrouter-thinking-1k" # Already verified
]

game_id = "ls20-016295f7601e"
results = {}

print(f"Starting verification for {len(models)} updated models...")

for model in models:
    print(f"\nTesting model: {model}")
    try:
        # Run with 1 action to verify connectivity and basic functioning
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        main_py = os.path.join(project_root, "main.py")
        cmd = [
            "uv", "run", "python", main_py,
            "--game_id", game_id,
            "--config", model,
            "--max_actions", "1"
        ]
        
        # Capture output to avoid cluttering, but print if failed
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            env=os.environ.copy()
        )
        
        if result.returncode == 0:
            print(f"✅ {model}: PASSED")
            results[model] = "PASSED"
        else:
            print(f"❌ {model}: FAILED")
            print(f"Error output:\n{result.stderr}")
            print(f"Standard output:\n{result.stdout}")
            results[model] = "FAILED"
            
    except Exception as e:
        print(f"❌ {model}: EXCEPTION - {e}")
        results[model] = f"EXCEPTION: {e}"

print("\n" + "="*30)
print("VERIFICATION RESULTS")
print("="*30)
for model, status in results.items():
    print(f"{model}: {status}")
print("="*30)

if all(status == "PASSED" for status in results.values()):
    print("\nAll updated Gemini models verified successfully!")
    sys.exit(0)
else:
    print("\nSome models failed verification.")
    sys.exit(1)
