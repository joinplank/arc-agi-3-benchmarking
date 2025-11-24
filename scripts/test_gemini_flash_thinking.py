
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

model_name = "gemini-2.5-flash"
print(f"Testing {model_name} with thinking config...")

try:
    response = client.models.generate_content(
        model=model_name,
        contents="Explain how a computer works in one sentence.",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            thinking_budget=1024
        )
    )
    print("Success!")
    print(response.text)
    if hasattr(response, 'usage_metadata'):
        print(f"Thinking tokens: {response.usage_metadata.thoughts_token_count}")
except Exception as e:
    print(f"Failed: {e}")
