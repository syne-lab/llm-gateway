import os

import openai

client = openai.OpenAI(
  base_url="http://127.0.0.1:31211/v1/",
  api_key=os.getenv("LLM_GATEWAY_TOKEN")
)

response = client.chat.completions.create(
    model="anthropic:claude-3-haiku-20240307",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    stream=True
)

for chunk in response:
    if hasattr(chunk, 'choices') and chunk.choices:
        message = chunk.choices[0].delta.content
        if message:
            print(message, end='', flush=True)

response = client.chat.completions.create(
    model="anthropic:claude-3-haiku-20240307",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    stream=False
)

print(response.choices[0].message.content)
