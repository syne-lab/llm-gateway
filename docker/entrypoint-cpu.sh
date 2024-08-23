#!/bin/bash

set -eu -o pipefail

source /opt/venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 31211 --workers 12
