import json
import urllib.request

try:
    with urllib.request.urlopen("http://172.31.208.1:11434/api/tags", timeout=2.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("Available models in Ollama on Windows host:")
        for m in data.get("models", []):
            print(f" - {m.get('name')}")
except Exception as e:
    print(f"Error querying Ollama: {e}")
