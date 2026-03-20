import asyncio
import functools
import random
import time
from collections.abc import Callable


def retry(
    func: Callable | None = None,
    errors: tuple = (Exception,),
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 10,
    max_delay: float = 60.0,
):
    """Retry a function with exponential backoff."""

    def decorator_retry(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specific errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        ) from e

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    # Cap the delay to max_delay
                    delay = min(delay, max_delay)

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    if func is None:
        return decorator_retry
    return decorator_retry(func)


def async_retry(
    func: Callable | None = None,
    errors: tuple = (Exception,),
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
):
    """Retry an async function with exponential backoff."""

    def decorator_retry(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return await func(*args, **kwargs)

                # Retry on specific errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        ) from e

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay (use asyncio.sleep for async)
                    await asyncio.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    if func is None:
        return decorator_retry
    return decorator_retry(func)
