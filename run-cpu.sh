#!/bin/bash

set -eux -o pipefail

find . -name "__pycache__" -type d -prune -exec rm -r '{}' +
docker rm -f $(docker ps -a -q -f "ancestor=szhan197/llm_gateway_cpu:latest") || true
docker rmi szhan197/llm_gateway_cpu:latest || true
docker build -f Dockerfile.cpu -t szhan197/llm_gateway_cpu:latest --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
docker network create llm-gateway || true
docker run -t -i -d \
    --net llm-gateway \
    -h llm-gateway \
    --name llm-gateway \
    szhan197/llm_gateway_cpu:latest
