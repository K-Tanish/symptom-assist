import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")
print(f"Testing key: {key[:10]}...")

try:
    client = Groq(api_key=key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "hi"}],
    )
    print("Success!")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
