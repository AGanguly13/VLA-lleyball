import requests, json, os

url   = "https://router.huggingface.co/v1/chat/completions"
model = "meta-llama/Llama-3.1-8B-Instruct"
key   = os.environ.get("LLAMA_INSTRUCT_API_KEY", "your-key-here")

# fake observation summary
state_summary = "Observation vector (33 values):\n" + \
    "\n".join(f"  [{i}]: {0.1*i:.3f}" for i in range(33))

payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": "You are a volleyball drone coach. Summarize the game state in 2 sentences."
        },
        {
            "role": "user",
            "content": f"Current game state:\n{state_summary}"
        }
    ],
    "max_tokens": 150,
    "temperature": 0.2,
}

resp = requests.post(
    url,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json=payload,
    timeout=10,
)
print(resp.json()["choices"][0]["message"])