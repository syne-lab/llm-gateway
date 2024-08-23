[[_TOC_]]

# Restricted Model Access Guideline

## First request access and then remember to set environment variables before run

### Hugging Face

```bash
 export HUGGING_FACE_HUB_TOKEN="" # Remember to add a space before the command to avoid it being recorded by the history.
```

### OpenAI

```bash
 export OPENAI_API_KEY="" # Remember to add a space before the command to avoid it being recorded by the history.
```

### Anthropic

```bash
 export ANTHROPIC_API_KEY="" # Remember to add a space before the command to avoid it being recorded by the history.
```

## Token creation

### Hugging Face

Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and click on `new token`

### OpenAI

Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) and click on `Create new secret key`

## Models

### Llama2

Go to [https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and complete the form. 

Email should be **the same as the Hugging Face account email**.

Remember to tick 

![](../../docs/assets/67f2e6a0.png)

Go to [https://huggingface.co/meta-llama/Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat)

![](../../docs/assets/958b6719.png)

Log in and then request access.

These series of models will all be available if you request once. No need for rerequest non-chat, 7b, 70b versions.

### Starcoder

Goto [https://huggingface.co/bigcode/starcoder](https://huggingface.co/bigcode/starcoder)

![](../../docs/assets/1e519173.png)

Log in and then request access.
