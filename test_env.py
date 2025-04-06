#!/usr/bin/env python
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Try to load .env with explicit path
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
print(f"Looking for .env file at: {env_path}")

if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded .env file from: {env_path}")
else:
    # Try to find .env file automatically
    env_file = find_dotenv()
    if env_file:
        load_dotenv(dotenv_path=env_file)
        print(f"Found and loaded .env file from: {env_file}")
    else:
        print("No .env file found!")

# Check key environment variables
required_vars = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "GROQ_API_KEY",
    "SERPER_API_KEY",
    "INDIANKANOON_API_TOKEN"
]

all_present = True
for var in required_vars:
    value = os.getenv(var)
    if value:
        # Mask the value for security
        masked = value[:6] + "..." + value[-4:] if len(value) > 10 else "****"
        print(f"✅ {var} is set: {masked}")
    else:
        print(f"❌ {var} is NOT set!")
        all_present = False

if all_present:
    print("\n✅ All required environment variables are set!")
else:
    print("\n❌ Some required environment variables are missing!")
    sys.exit(1)
