import httpx
from assume.config import VERTEX_API_KEY, VERTEX_PROJECT_ID, VERTEX_MODEL

payload = {"contents": [{"role": "user", "parts": [{"text": "Hello, respond with OK"}]}]}

url = f"https://aiplatform.googleapis.com/v1/projects/{VERTEX_PROJECT_ID}/locations/global/publishers/google/models/{VERTEX_MODEL}:generateContent"
headers = {"Content-Type": "application/json", "x-goog-api-key": VERTEX_API_KEY}

print(f"Connecting to {url}...")
try:
    resp = httpx.post(url, headers=headers, json=payload, timeout=15)
    print("Vertex status:", resp.status_code)
    print("Vertex response:", resp.text[:500])
except Exception as e:
    print("Exception:", e)
