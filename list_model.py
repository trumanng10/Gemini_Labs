import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# List all available models
print("Available models:")
for model in genai.list_models():
    print(f"- {model.name}")
    print(f"  Supported methods: {model.supported_generation_methods}")
    print()
