#!/bin/bash

set -eu -o pipefail

source /opt/venv/bin/activate
# Change to 0.0.0.0 when pushing to docker hub
uvicorn server:app --host 0.0.0.0 --port 31211 2>&1
