# test_openai.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test 1: Check API key
api_key = os.environ.get("OPENAI_API_KEY")
print(f"API Key loaded: {bool(api_key)}")

# Test 2: Check OpenAI import and version
try:
    from openai import OpenAI
    print(f"OpenAI imported successfully")
    
    # Test 3: Create client
    client = OpenAI(api_key=api_key)
    print(f"Client created: {client}")
    
    # Test 4: Check if chat attribute exists
    print(f"Has 'chat' attribute: {hasattr(client, 'chat')}")
    if hasattr(client, 'chat'):
        print(f"Chat type: {type(client.chat)}")
        print(f"Has 'completions': {hasattr(client.chat, 'completions')}")
    
    # Test 5: Try a simple API call
    print("Testing API call...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'Hello World'"}],
        max_tokens=10
    )
    print(f"API call successful: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()