# Load model directly
import copy
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    logging,
    StoppingCriteriaList,
    StoppingCriteria,
    GenerationConfig,
)
from typing import Dict, List, Optional, Union, Iterator
from threading import Thread

from llm_gateway.llmabc import LLMAbstractBaseClass
from llm_gateway.llms import LLMs, Message, LLMsPromptTemplate, TransformersModelInfo


class StoppingCriteriaSub(StoppingCriteria):
    """@private"""

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


class TransformersLLM(LLMAbstractBaseClass):
    def __init__(self, model: LLMs, tokens: Optional[Dict[str, str]] = None):
        super().__init__(model, tokens)
        logging.set_verbosity_error()
        model_info: TransformersModelInfo = self.model.getInfo()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model.value,
            trust_remote_code=True,
            token=tokens.get("HUGGING_FACE_HUB_TOKEN"),
            revision=model_info.revision,
        )
        self._llm = AutoModelForCausalLM.from_pretrained(
            model.value,
            trust_remote_code=True,
            token=tokens.get("HUGGING_FACE_HUB_TOKEN"),
            revision=model_info.revision,
        )
        self._base_config = GenerationConfig.from_pretrained(
            model.value,
            token=tokens.get("HUGGING_FACE_HUB_TOKEN"),
            revision=model_info.revision,
        )
        self._template = LLMsPromptTemplate[model_info.template]

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
        stop = [stop] if isinstance(stop, str) else stop
        stop_words_ids = [
            self._tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in stop
        ]
        config = copy.deepcopy(self._base_config)
        config.update(
            max_new_tokens=max_tokens,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            ),
            temperature=temperature,
            top_p=top_p,
            diversity_penalty=frequency_penalty,
            repetition_penalty=repeat_penalty,
            top_k=top_k,
        )
        generated_ids = self._llm.generate(
            inputs=self._tokenizer(prompt, return_tensors="pt")["input_ids"],
            generation_config=config,
        )
        output = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        for stop_word in stop:
            index = output.rfind(stop_word)
            if index != -1 and index + len(stop_word) == len(output):
                output = output[:index]
                break
        return output

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
        stop = [stop] if isinstance(stop, str) else stop
        stop_words_ids = [
            self._tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in stop
        ]
        streamer = TextIteratorStreamer(
            self._tokenizer, timeout=15.0, skip_prompt=True, skip_special_tokens=True
        )
        config = copy.deepcopy(self._base_config)
        config.update(
            max_new_tokens=max_tokens,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            ),
            temperature=temperature,
            top_p=top_p,
            diversity_penalty=frequency_penalty,
            repetition_penalty=repeat_penalty,
            top_k=top_k,
        )
        generate_kwargs = dict(
            inputs=self._tokenizer(prompt, return_tensors="pt").input_ids,
            generation_config=config,
            streamer=streamer,
        )
        t = Thread(target=self._llm.generate, kwargs=generate_kwargs)
        t.start()
        for output in streamer:
            if output in stop:
                continue
            yield output

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
    ) -> Iterator[Message]:
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
