import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

models = genai.list_models()

for m in models:
    print(m.name, " -> ", m.supported_generation_methods)