from llm_gateway import LLM, LLMs

llm = LLM(LLMs.Yi_34b_Chat)
llm_session = llm.create_session()
sys_prompt = "Output a number only. Don't include the question in your response"
user_prompt = (
    "Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?"
)
llm_session.temperature(0.5).max_tokens(256).stop("<|im_end|>")
stream = llm_session.stream_inference(
    [
        {
            "role": "system",
            "content": sys_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
)
print(user_prompt)
for output in stream:
    print(output["content"], end="", flush=True)
print()
