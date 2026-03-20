import os
from urllib.parse import urlparse

import uvicorn

from llm_gateway.router import create_app

app = create_app()


def main():
    url = urlparse(os.getenv("LLM_GATEWAY_ENDPOINT", "http://127.0.0.1:31211"))
    uvicorn.run(
        "__main__:app",
        host=url.hostname or "127.0.0.1",
        port=url.port or 31211,
        workers=8,
    )


if __name__ == "__main__":
    main()
