from llm_gateway import LLM, LLMs

llm = LLM(LLMs.Llama_2_7b)
llm_session = llm.create_session()
prompt = {
    "role": None,
    "content": """def fib(n):\n"""
}
llm_session.temperature(1.1).max_tokens(256).stop("<EOT>")
stream = llm_session.stream_inference([prompt])
print(prompt["content"], end="")
for output in stream:
    print(output["content"], end="", flush=True)
print()
