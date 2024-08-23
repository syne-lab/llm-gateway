import asyncio
import json
from typing import Dict, List, Optional, Union
from typing_extensions import TypedDict
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
import torch
from llm_gateway import LLM, LLMs
from llm_gateway.llmabc import LLMAbstractBaseClass
from llm_gateway.llms import Message
from pydantic import BaseModel
import gc
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import os
from hashlib import blake2b
import functools

@functools.cache
def get_expected_token_hash():
    return blake2b(os.getenv("LLM_GATEWAY_TOKEN").encode()).hexdigest() if os.getenv("LLM_GATEWAY_TOKEN") else None

# We will handle a missing token ourselves
get_bearer_token = HTTPBearer(auto_error=False)

async def get_token(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    expected_token_hash = get_expected_token_hash()
    if expected_token_hash is None:
        print("Warning: LLM_GATEWAY_TOKEN is not set, anyone can access the API")
        return ''

    if auth is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )
    token = auth.credentials
    if blake2b(token.encode()).hexdigest() != expected_token_hash:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    return token

class ModelLastUsed(TypedDict):
    model_name: str
    llm: LLMAbstractBaseClass

app = FastAPI()
app.model_last_used = None
app.lock = asyncio.Lock()


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    print(f"HTTP error: {repr(exc)}")
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


class CompParam(BaseModel):
    model: str
    prompt: List[Message]
    suffix: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = []
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.1
    top_k: int = 40
    tokens: Optional[Dict[str, str]] = None


@app.post("/inference")
async def create_inference(param: CompParam, token: str = Depends(get_token)):
    query_result = LLMs.new_query().where_name_is(param.model).exec()
    if len(query_result) == 0:
        raise StarletteHTTPException(status_code=404, detail="Model not found")
    query_result = query_result[0]
    await app.lock.acquire()
    try:
        llm = None
        if app.model_last_used is None or app.model_last_used["model_name"] != param.model:
            app.model_last_used = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            llm = LLM(query_result.name, param.tokens)
            app.model_last_used = {"model_name": param.model, "llm": llm}
        else:
            llm = app.model_last_used["llm"]
        if (query_result.price > 0):
            app.model_last_used = None
        result = llm.inference(
            param.prompt,
            param.suffix,
            param.max_tokens,
            param.temperature,
            param.top_p,
            param.logprobs,
            param.echo,
            param.stop,
            param.frequency_penalty,
            param.presence_penalty,
            param.repeat_penalty,
            param.top_k,
        )
        return result
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        raise StarletteHTTPException(status_code=500, detail=str(e))
    finally:
        app.lock.release()


@app.post("/stream_inference")
async def create_stream_inference(param: CompParam, token: str = Depends(get_token)):
    query_result = LLMs.new_query().where_name_is(param.model).exec()
    if len(query_result) == 0:
        raise StarletteHTTPException(status_code=404, detail="Model not found")
    query_result = query_result[0]
    await app.lock.acquire()
    try:
        llm = None
        if app.model_last_used is None or app.model_last_used["model_name"] != param.model:
            app.model_last_used = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            llm = LLM(query_result.name, param.tokens)
            app.model_last_used = {"model_name": param.model, "llm": llm}
        else:
            llm = app.model_last_used["llm"]
        if (query_result.price > 0):
            app.model_last_used = None
        stream = json_stream_wrapper(app, llm, param)
        return StreamingResponse(stream)
    except KeyboardInterrupt:
        app.lock.release()
        raise KeyboardInterrupt
    except Exception as e:
        app.lock.release()
        raise StarletteHTTPException(status_code=500, detail=str(e))

async def json_stream_wrapper(app, llm: LLMAbstractBaseClass, param: CompParam):
    try:
        for output in llm.stream_inference(
                param.prompt,
                param.suffix,
                param.max_tokens,
                param.temperature,
                param.top_p,
                param.logprobs,
                param.echo,
                param.stop,
                param.frequency_penalty,
                param.presence_penalty,
                param.repeat_penalty,
                param.top_k,
            ):
                # Give time to response to the client
                await asyncio.sleep(0.03)
                yield json.dumps(output) + "\n"
    finally:
        app.lock.release()
