from llm_gateway import LLM, LLMs

llm = LLM(LLMs.StarCoder)
llm_session = llm.create_session()
prompt = {
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
# starcoder needs a relative low temperature to prevent some random output.
llm_session.temperature(0.2).max_tokens(256)
stream = llm_session.stream_inference([prompt])
print(prompt["content"], end="")
for output in stream:
    print(output["content"], end="", flush=True)
print()
