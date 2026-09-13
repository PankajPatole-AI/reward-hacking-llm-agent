import os
import cohere
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

client = cohere.ClientV2(api_key)

response = client.chat(
    model="command-a-plus-05-2026",
    messages=[
        {
            "role": "user",
            "content": "Choose one action: UP, DOWN, LEFT, RIGHT. Respond with only the action."
        }
    ]
)

for item in response.message.content:
    if hasattr(item, "text"):
        print(item.text)