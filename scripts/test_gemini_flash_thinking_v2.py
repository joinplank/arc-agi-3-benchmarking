
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not found")
    exit(1)

client = genai.Client(api_key=api_key)

models_to_test = [
    "gemini-2.0-flash-thinking-exp-01-21", # Control: Should work
    "gemini-2.5-flash" # Test: Does it support thinking?
]

for model_name in models_to_test:
    print(f"\nTesting {model_name} with thinking config...")
    try:
        # Using the structure from models.yml (passing dict to thinking_config)
        # Note: models.yml has thinking_budget inside thinking_config
        response = client.models.generate_content(
            model=model_name,
            contents="Explain how a computer works in one sentence.",
            config=types.GenerateContentConfig(
                thinking_config={"include_thoughts": True, "thinking_budget": 1024}
            )
        )
        print(f"✅ Success for {model_name}!")
        if hasattr(response, 'usage_metadata'):
            print(f"Thinking tokens: {response.usage_metadata.thoughts_token_count}")
    except Exception as e:
        print(f"❌ Failed for {model_name}: {e}")
