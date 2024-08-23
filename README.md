# LLM Gateway

## Client library for the LLM Gateway

There is also a client library available to interact with this gateway. It's especially useful if you want yo run gpu models on a remote server.

The client library is available at [llm-gateway-client](https://github.com/syne-lab/llm-gateway-client).

## Install as a package

### Pip

```sh
pip install git+https://github.com/syne-lab/llm-gateway
```

### Poetry

```sh
poetry add git+https://github.com/syne-lab/llm-gateway
```

### Rye

```sh
UV_INDEX_STRATEGY=unsafe-any-match rye add llm-gateway --git https://github.com/syne-lab/llm-gateway
```

### Set the auth tokens

Some models are restricted and not available unless you accept the license and then set a token:

```sh
# Remember to add a space before the command to avoid it being recorded by the history.
 export HUGGING_FACE_HUB_TOKEN=""
 export OPENAI_API_KEY=""
 export ANTHROPIC_API_KEY=""
```

## Special infilling tokens you need to know (even with preset templates)

When interacting with code completion models which often have infilling mode,
you need to add a token `<FILL>` or `<SUFFIX-FIRST-FILL>` into the prompt.

Reference: https://arxiv.org/abs/2207.14255

Example:

```python
from llm_gateway import LLM, LLMs

llm = LLM(LLMs.CodeLlama_13b)
llm_session = llm.create_session()
prefix = """float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
"""

suffix= """
	return y;
}
"""
llm_session.temperature(1.1)
llm_session.max_tokens(2048)
llm_session.stop("<EOT>")
stream = llm_session.stream_inference([
    {
        "role": None,
        "content": prefix + "<FILL>" + suffix
	}
])
print(prefix, end="")
for output in stream:
    print(output["content"], end="")
print()
print(suffix)
```

## Examples

Examples can be found in the test folder.

## Core APIs

### LLMs

```python
class LLMs(Enum):
    Llama_2_7b = "meta-llama/Llama-2-7b-hf"  # Need HF token
    Llama_2_13b = "meta-llama/Llama-2-13b-hf"  # Need HF token
    Llama_2_70b_Quant = "TheBloke/Llama-2-70B-GGUF"
    Llama_2_7b_Chat = "meta-llama/Llama-2-7b-chat-hf"  # Need HF token
    Llama_2_13b_Chat = "meta-llama/Llama-2-13b-chat-hf"  # Need HF token
    Llama_2_70b_Chat_Quant = "TheBloke/Llama-2-70B-Chat-GGUF"
    CodeLlama_7b = "codellama/CodeLlama-7b-hf"
    CodeLlama_13b = "codellama/CodeLlama-13b-hf"
    CodeLlama_34b = "codellama/CodeLlama-34b-hf"
    CodeLlama_34b_Quant = "TheBloke/CodeLlama-34B-GGUF"
    CodeLlama_7b_Instruct = "codellama/CodeLlama-7b-Instruct-hf"
    CodeLlama_13b_Instruct = "codellama/CodeLlama-13b-Instruct-hf"
    CodeLlama_34b_Instruct = "codellama/CodeLlama-34b-Instruct-hf"
    CodeLlama_34b_Instruct_Quant = "TheBloke/CodeLlama-34B-Instruct-GGUF"
    Llama_3_8b_Instruct = "meta-llama/Meta-Llama-3-8B-Instruct"  # Need HF token
    Llama_3_70b_Instruct_Quant = "QuantFactory/Meta-Llama-3-70B-Instruct-GGUF"
    StarCoder = "bigcode/starcoder"  # Need HF token
    Mistral_7b_0_1 = "TheBloke/Mistral-7B-v0.1-GGUF"
    Mistral_7b_0_1_Instruct = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    Mixtral_8x7b_0_1 = "TheBloke/Mixtral-8x7B-v0.1-GGUF"
    Mixtral_8x7b_0_1_Instruct = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
    Yi_6b = "01-ai/Yi-6B"
    Yi_34b = "01-ai/Yi-34B"
    Yi_6b_200k = "01-ai/Yi-6B-200K"
    Yi_34b_200k = "01-ai/Yi-34B-200K"
    Yi_6b_Chat = "01-ai/Yi-6B-Chat"
    Yi_34b_Chat = "01-ai/Yi-34B-Chat"
    Phi_3_Mini_128k_Instruct = "microsoft/Phi-3-mini-128k-instruct"
    CodeGen25_7b_Multi = "Salesforce/codegen25-7b-multi"
    GPT4_Turbo = "gpt-4-turbo"  # Need OpenAI token
    GPT4_Turbo_2024_0409 = "gpt-4-turbo-2024-04-09"  # Need OpenAI token
    GPT4_0125_Preview = "gpt-4-0125-preview" # Need OpenAI token
    GPT4_1106_Preview = "gpt-4-1106-preview" # Need OpenAI token
    GPT4_Turbo_Preview = "gpt-4-turbo-preview" # Need OpenAI token
    GPT4 = "gpt-4"  # Need OpenAI token
    GPT4_32k = "gpt-4-32k"  # Need OpenAI token
    GPT3_5_Turbo = "gpt-3.5-turbo"  # Need OpenAI token
    Claude_3_opus_20240229 = "claude-3-opus-20240229" # Need Anthropic token
    Claude_3_sonnet_20240229 = "claude-3-sonnet-20240229" # Need Anthropic token
    Claude_3_haiku_20240307 = "claude-3-haiku-20240307" # Need Anthropic token
    Claude_2_1 = "claude-2.1"  # Need Anthropic token
    Claude_2_0 = "claude-2.0"  # Need Anthropic token
    Claude_Instant_1_2 = "claude-instant-1.2"  # Need Anthropic token
```

Example:

```python
from llm_gateway import LLM, LLMs

llm = LLM(LLMs.CodeLlama_13b_Instruct)
```

#### Query LLMs

```python
result = (
    LLMs.new_query()
    .where_expert_in({"general"})
    .where_chat_is(True)
    .where_infill_is(False)
    .order_by_score()
    .limit(2)
    .exec()
)

print(result)
```

### Cache

```python
from llm_gateway import set_model_cache_dir

set_model_cache_dir("/path/to/cache/dir")
```

### LLM

```python
def LLM(model: LLMs) -> LLMAbstractBaseClass
```

Example:

```python
from llm_gateway import LLM, LLMs

llm = LLM(LLMs.CodeLlama_13b_Instruct)
```

### LLMAbstractBaseClass

#### Methods:

```python
def create_session(self) -> LLMSession
```

Create an LLMSession for this model.

### Message

```python
class Message(TypedDict):
    role: Optional[Union[Literal["system"], Literal["user"], Literal["assistant"]]]
    content: str
```

Example:

```python
{
    "role": "user",
    "content": "Hello, world!"
}
```

### LLMSession

Each LLMSession can have different model parameters, and different contexts.

LLMSession will record the given prompts and model outputs for the next query.

#### Methods:

```python
def clear_context(self)
```

Clear the dialogue context.

---

```python
def *(self) -> LLMSession
```

Set model parameters for current session.

---

```python
def raw_inference(self, prompt: str) -> str
```

Interact with the model without a preset template.

Example:

````python
from llm_gateway import LLM, LLMs

llm = LLM(LLMs.CodeLlama_13b_Instruct)
llm_session = llm.create_session()
prompt = """
[INST]
<<SYS>>
You are a security bot writing c programs, aiming to find potential bugs inside the API.
<</SYS>>

Write a complex c program to fuzz following function:
```c
#include <sys/socket.h>

int socket(int domain, int type, int protocol);
```
[/INST]
"""

output = llm_session.raw_inference(prompt)
print(output)
````

---

```python
def raw_stream_inference(self, prompt: str) -> Iterator[str]
```

Interact with the model without a preset template in a stream way.

Example:

````python
from llm_gateway import LLM, LLMs

llm = LLM(LLMs.CodeLlama_13b_Instruct)
llm_session = llm.create_session()
prompt = """
[INST]
<<SYS>>
You are a security bot writing c programs, aiming to find potential bugs inside the API.
<</SYS>>

Write a complex c program to fuzz following function:
```c
#include <sys/socket.h>

int socket(int domain, int type, int protocol);
```
[/INST]
"""

stream = llm_session.raw_stream_inference(prompt)
for output in stream:
    print(output, end="")
print()
````

---

```python
def inference(
        self,
        messages: List[Message],
    ) -> Message
```

Interact with the model with a preset template.

Example:

````python
from llm_gateway import LLM, LLMs

llm = LLM(LLMs.CodeLlama_13b_Instruct)
llm_session = llm.create_session()
sys_prompt = """
You are a security bot writing c programs, aiming to find potential bugs inside the API.
"""

user_prompt = """
Write a complex c program to fuzz following function:
```c
#include <sys/socket.h>

int socket(int domain, int type, int protocol);
```
"""

output = llm_session.inference([
    {
        "role": "system",
        "content": sys_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    },
])
print(output["content"])
llm_session.clear_context()
output = llm_session.inference([
    {
        "role": "user",
        "content": user_prompt
    },
])
print(output["content"])
````

---

```python
def stream_inference(
        self,
        messages: List[Message],
    ) -> Iterator[Message]
```

Interact with the model with a preset template in a stream way.

Example:

````python
from llm_gateway import LLM, LLMs

llm = LLM(LLMs.CodeLlama_13b_Instruct)
llm_session = llm.create_session()
sys_prompt = """
You are a security bot writing c programs, aiming to find potential bugs inside the API.
Follow the user's requirements carefully & to the letter.
Output only the code in a single code block.
Use Markdown formatting in your answers.
Make sure to include the programming language name at the start of the Markdown code blocks.
"""

user_prompt= """
You can include any complex operations related to socket programming as long as they are posix systemcalls.
Make sure you write the program in the main function.
Write a complex c program to fuzz following function:
```c
#include <sys/socket.h>

int socket(int domain, int type, int protocol);
```
"""
llm_session.temperature(1.1)
llm_session.max_tokens(2048)
llm_session.stop(["<<SYS>>", "<</SYS>>", "[INST]", "[/INST]"])
stream = llm_session.stream_inference([
    {
        "role": "system",
        "content": sys_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    },
    # The message below is generated by the codellama instruct, but due to the max_tokens limit,
    # it cuts off the message. Adding the generated message below will allow the codellama instruct
    # to complete the message.
    {
        "role": "assistant",
        "content": """
```c
/* Fuzzing the socket() function */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(void) {
    /* Create a new socket */
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    
    /* Set the address of the server */
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(struct sockaddr_in));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(80);
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
    
    /* Fuzz the domain parameter */
    for (int i=0; i<4; i++) {
        int domain = rand() % 3;
        if (socket(domain, SOCK_STREAM, 0) == -1) {
            perror("socket");
        } else {
            printf("[%d] Socket created successfully\n", domain);
        }
    }
    
    /* Fuzz the type parameter */"""
    }
])
for output in stream:
    print(output["content"], end="", flush=True)
print()
````

## Development

This project uses [Rye](https://rye-up.com/) for python project management.

### Install Rye

```sh
curl -sSf https://rye-up.com/get | bash
```

This will install Rye in your home directory.

### Install dependencies

```sh
UV_INDEX_STRATEGY=unsafe-any-match rye sync
```

Here, we set `UV_INDEX_STRATEGY=unsafe-any-match` to make the behavior of the `rye sync` the same as `pip install -r requirements.lock`. It's a workaround for custom index URLs. See issue https://github.com/astral-sh/uv/issues/171 for more details.
