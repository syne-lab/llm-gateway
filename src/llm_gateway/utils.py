# define a retry decorator
import random
import time
import openai
import anthropic


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 2,
    errors: tuple = (openai.RateLimitError, anthropic.RateLimitError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            except (openai.APIStatusError, anthropic.APIStatusError) as e:
                if e.status_code >= 500:
                    # Increment the delay
                    if delay < 600:
                        delay *= exponential_base * (1 + jitter * random.random())
                    time.sleep(delay)
                else:
                    raise e

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper