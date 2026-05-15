from groq import Groq
import os
from dotenv import load_dotenv

# .env load karo
from pathlib import Path
load_dotenv(dotenv_path=Path(".env"))

api_key = os.getenv("GROQ_API_KEY")
print("API KEY:", api_key)

client = Groq(api_key=api_key)

try:
    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("SUCCESS ✅")
    print(res.choices[0].message.content)

except Exception as e:
    print("ERROR ❌:", e)