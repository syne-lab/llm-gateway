from typing import List
from llm_gateway import LLM, LLMs, Message
from llm_gateway.llms import mistral_style_instruct_prompt_template

prompts_1: List[Message] = [
    {
        "role": "user",
        "content": """Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?"""
    },
    {
        "role": "assistant",
        "content": "5"
    }
]

prompts_2: List[Message] = [
    {
        "role": "system",
        "content": "Answer the following questions:"
    },
    {
        "role": "user",
        "content": """Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?"""
    },
    {
        "role": "assistant",
        "content": "5"
    },
    {
        "role": "system",
        "content": "Answer the following questions:"
    },
    {
        "role": "user",
        "content": """Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?"""
    },
]

assert mistral_style_instruct_prompt_template(prompts_1) == """<s>[INST]
Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?
[/INST]
5</s>"""

assert mistral_style_instruct_prompt_template(prompts_2) == """<s>[INST]
Answer the following questions:
Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?
[/INST]
5</s>
[INST]
Answer the following questions:
Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?
[/INST]
"""

llm = LLM(LLMs.Mixtral_8x7b_0_1_Instruct)
llm_session = llm.create_session()
prompt: Message = {
    "role": "user",
    "content": """Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?"""
}
llm_session.temperature(0.5).max_tokens(256).stop(["\n"])
stream = llm_session.stream_inference([prompt])
print(prompt["content"], end="")
for output in stream:
    print(output["content"], end="", flush=True)
print()