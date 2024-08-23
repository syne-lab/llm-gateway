from llm_gateway import LLM, LLMs, Message

llm = LLM(LLMs.Phi_3_Mini_128k_Instruct)
llm_session = llm.create_session()
prompt: Message = {
    "role": "user",
    "content": """Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?"""
}
llm_session.temperature(0.1).max_tokens(256)
stream = llm_session.stream_inference([prompt])
print(prompt["content"], end="\n")
for output in stream:
    print(output["content"], end="", flush=True)
print()