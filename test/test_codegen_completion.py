from llm_gateway import LLM, LLMs, Message

llm = LLM(LLMs.CodeGen25_7b_Multi)
llm_session = llm.create_session()
prompt: Message = {
    "role": None,
    "content": """float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
"""
}
llm_session.temperature(1.1).max_tokens(256).stop(["}\n\n", "<|endoftext|>"])
stream = llm_session.stream_inference([prompt])
print(prompt["content"], end="")
for output in stream:
    print(output["content"], end="", flush=True)
print()
