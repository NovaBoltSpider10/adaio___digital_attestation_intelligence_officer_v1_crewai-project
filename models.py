import requests
import os

print("API_KEY:", os.environ.get("GROQ_API_KEY"))
api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {"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"}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())