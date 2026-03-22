# LLM Gateway

## Client library for the LLM Gateway

Use the `openai` and set base URL to `http://host:port/v1/` to use the LLM Gateway.

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:31211/v1/",
    api_key=os.getenv("LLM_GATEWAY_TOKEN"),
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
```

## Install as a package

### Pip

```sh
pip install 'llm-gateway @ git+https://github.com/syne-lab/llm-gateway'
```

### Poetry

```sh
poetry add 'git+https://github.com/syne-lab/llm-gateway'
```

### uv

```sh
uv add llm-gateway --git https://github.com/syne-lab/llm-gateway
```

### Set the auth tokens

Some models are restricted and not available unless you accept the license and then set a token:

```sh
# Remember to add a space before the command to avoid it being recorded by the history.
 export HUGGING_FACE_HUB_TOKEN=""
 export OPENAI_API_KEY=""
 export ANTHROPIC_API_KEY=""
```

## Examples

Examples can be found in the test folder.

## Development

This project uses [UV](https://docs.astral.sh/uv/) for python project management.

### Install UV

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This will install Rye in your home directory.

### Install dependencies

```sh
uv sync
```

## Deploy on GPU Instance

### Set the auth tokens for Gateway

```sh
export LLM_GATEWAY_TOKEN=""
```
