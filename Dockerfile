FROM ubuntu:noble

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/home/ubuntu/llm_gateway/src

RUN apt update && \
    apt install -y wget git build-essential python3 libpython3-dev python3-venv curl neovim

COPY requirements.lock /tmp/requirements.lock

RUN sed -i '/^-e/d' /tmp/requirements.lock && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.lock

USER ubuntu

COPY --chown=ubuntu:ubuntu . /home/ubuntu/llm_gateway

WORKDIR /home/ubuntu/llm_gateway

ENTRYPOINT ["/opt/venv/bin/python3", "src/llm_gateway/main.py"]
