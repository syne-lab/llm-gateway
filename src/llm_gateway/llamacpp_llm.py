from __future__ import annotations
from contextlib import contextmanager
import fnmatch
import os
import subprocess
from huggingface_hub import snapshot_download
from typing import Dict, List, Optional, Union, Iterator

import torch
from llama_cpp import Llama

from llm_gateway.llms import LLMs, LLMsPromptTemplate, LlamaCppModelInfo, Message
from llm_gateway.llmabc import LLMAbstractBaseClass

_gateway_dir = os.path.join(os.path.expanduser("~"), ".cache", "llm_gateway")


_model_convert_script_map = {
    "default": "python llama.cpp/convert.py {}",
    "hf": "python llama.cpp/convert-hf-to-gguf.py {}",
}


def convert_gguf(model_info: LlamaCppModelInfo, model_path):
    """@private"""
    script = _model_convert_script_map[model_info.gguf_script]
    with cd(_gateway_dir):
        sh(script.format(model_path))


@contextmanager
def cd(path):
    """@private"""
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def sh(command):
    """@private"""
    try:
        if isinstance(command, list):
            command = " ".join(command)
        return (
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            .decode("utf-8")
            .rstrip()
        )
    except KeyboardInterrupt:
        exit(-1)
    except Exception:
        return None


class LlamaCppLLM(LLMAbstractBaseClass):
    def __init__(self, model: LLMs, tokens: Optional[Dict[str, str]] = None) -> None:
        super().__init__(model, tokens)
        os.makedirs(_gateway_dir, 755, exist_ok=True)
        llamacpp_hash = "6ecf3189e00a1e8e737a78b6d10e1d7006e050a2"
        if not os.path.exists(os.path.join(_gateway_dir, "llama.cpp")):
            print(f"Cloning the llama.cpp {llamacpp_hash}")
            with cd(_gateway_dir):
                sh("git clone https://github.com/ggerganov/llama.cpp")
                sh(f"git checkout {llamacpp_hash}")
        else:
            with cd(os.path.join(_gateway_dir, "llama.cpp")):
                if sh("git rev-parse HEAD") != llamacpp_hash:
                    print(f"Updating the llama.cpp {llamacpp_hash}")
                    sh("git fetch origin master:master")
                    sh(f"git checkout {llamacpp_hash}")
        model_info: LlamaCppModelInfo = self.model.getInfo()
        self._template = LLMsPromptTemplate[model_info.template]
        self._model_path = os.path.join(_gateway_dir, "models", model.value)
        snapshot_download(
            model.value,
            local_dir=self._model_path,
            allow_patterns=model_info.file_list,
            token=tokens.get("HUGGING_FACE_HUB_TOKEN"),
            revision=model_info.revision,
        )
        print("Downloaded model: ", model.name, "to", self._model_path)
        self._gguf_path = ""
        for file in os.listdir(self._model_path):
            if fnmatch.fnmatch(file, "*-000??-of-000??.gguf"):
                # https://github.com/ggerganov/llama.cpp/discussions/6404
                if not fnmatch.fnmatch(file, "*-00001-of-000??.gguf"):
                    continue
                self._gguf_path = os.path.join(self._model_path, file)
                break
            if fnmatch.fnmatch(file, "*.gguf"):
                self._gguf_path = os.path.join(self._model_path, file)
                break
            if fnmatch.fnmatch(file, "*.gguf-split-?"):
                file_name = file.split(".")[0]
                quantization = file.split(".")[1]
                self._gguf_path = os.path.join(self._model_path, f"{file_name}.{quantization}.gguf")
                if not os.path.exists(self._gguf_path):
                    print("Merging split gguf files")
                    with cd(self._model_path):
                        sh(
                            f"cat {file_name}.{quantization}.gguf-split-* > {file_name}.{quantization}.gguf"
                        )
                assert os.path.exists(self._gguf_path), "Failed to merge split gguf files"
                break
        if self._gguf_path == "":
            print("Converting model: ", model.name, "to .gguf format.")
            convert_gguf(model_info, self._model_path)
            print("Model converted to .gguf format.")
            for file in os.listdir(self._model_path):
                if fnmatch.fnmatch(file, "*.gguf"):
                    self._gguf_path = os.path.join(self._model_path, file)
                    break
        if self._gguf_path == "" or not os.path.exists(self._gguf_path):
            raise ValueError("GGUF Model conversion failed")
        self.max_context = model_info.max_context
        if torch.cuda.is_available():
            print("CUDA is available, using GPU")
            self._llm = Llama(
                model_path=self._gguf_path,
                n_ctx=self.max_context,
                verbose=True,
                n_gpu_layers=-1,
            )
            print("cuBLAS backend")
        else:
            self._llm = Llama(
                model_path=self._gguf_path, n_ctx=self.max_context, verbose=False
            )
            print("CPU backend")

    def raw_inference(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ) -> str:
        return self._llm(
            prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=False,
        )["choices"][0]["text"]

    def raw_stream_inference(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ) -> Iterator[str]:
        for output in self._llm(
            prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=True,
        ):
            yield output["choices"][0]["text"]

    def inference(
        self,
        prompt: List[Message],
        suffix: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: int | None = None,
        echo: bool = False,
        stop: str | List[str] | None = ...,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ) -> Message:
        raw_prompt = self._template(prompt)
        output = self.raw_inference(
            raw_prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
        )
        return {"role": "assistant", "content": output}

    def stream_inference(
        self,
        prompt: List[Message],
        suffix: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ):
        raw_prompt = self._template(prompt)
        for output in self.raw_stream_inference(
            raw_prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
        ):
            yield {"role": "assistant", "content": output}
