import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR: Could not find GOOGLE_API_KEY in .env file.")
    exit()

print("Connecting to Google's REST API...")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("\n✅ SUCCESS! Here are the exact embedding model names your key supports:\n")
    
    found_models = False
    for model in data.get("models", []):
        if "embedContent" in model.get("supportedGenerationMethods", []):
            print(f"  -> {model.get('name')}")
            found_models = True
            
    if not found_models:
        print("  [No embedding models found for this key]")
else:
    print("❌ API REJECTED THE REQUEST:")
    print(response.text)