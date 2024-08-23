from llm_gateway import LLM, LLMs

llm = LLM(LLMs.Llama_3_70b_Instruct_Quant)
llm_session = llm.create_session()
prompt = {
    "role": "user",
    "content": """Write function `fibonacci(n: int) -> int` in python. Output the program only. Minimize any other prose."""
}
llm_session.temperature(1.1).max_tokens(256)
stream = llm_session.stream_inference([prompt])
print(prompt["content"], end="")
for output in stream:
    print(output["content"], end="", flush=True)
print()
