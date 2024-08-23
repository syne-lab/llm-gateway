from llm_gateway import LLM, LLMs

llm = LLM(LLMs.GPT3_5_Turbo)
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
llm_session.temperature(1.1).max_tokens(512).stop(["}\n\n"])
stream = llm_session.stream_inference([
    {
        "role": "system",
        "content": sys_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    },
])
for output in stream:
    print(output["content"], end="", flush=True)
print()