import os

import openai

client = openai.OpenAI(
  base_url="http://127.0.0.1:31211/v1/",
  api_key=os.getenv("LLM_GATEWAY_TOKEN")
)

history = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
]

response = client.chat.completions.create(
    model="openai:gpt-4.1-nano",
    messages=history,
    stream=False
)

history.append(response.choices[0].message)

history.append(
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
)

response = client.chat.completions.create(
    model="openai:gpt-5-nano",
    messages=history,
    stream=False
)

print(response.choices[0].message.content)

history.append(response.choices[0].message)

history.append(
    {
        "role": "user",
        "content": "What is the capital of Germany?"
    }
)

response = client.chat.completions.create(
    model="openai:gpt-4.1-nano",
    messages=history,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)


