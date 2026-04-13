"""
hello_groq.py
A 30-second sanity check that your Groq API key works.
Run this FIRST before anything else. If this works, you're set.

Usage:
    python src/hello_groq.py
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit(
        "GROQ_API_KEY not found. Copy .env.example to .env and add your key."
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "Say hi to a developer named Docsy in one sentence."},
    ],
)

print("\n✅ Groq API works!\n")
print("Model said:", response.choices[0].message.content)
