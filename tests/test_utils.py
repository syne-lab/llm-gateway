import asyncio
import unittest
from unittest.mock import patch

from llm_gateway.utils import async_retry, retry


class TestRetryDecorator(unittest.TestCase):
    """Test cases for the synchronous retry decorator."""

    def test_retry_success_on_first_attempt(self):
        """Test that retry works when function succeeds on first attempt."""
        call_count = 0

        @retry(max_retries=3)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)

    def test_retry_success_after_failures(self):
        """Test that retry works when function succeeds after some failures."""
        call_count = 0

        @retry(max_retries=3, initial_delay=0.01)
        def eventually_successful_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = eventually_successful_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

    def test_retry_max_retries_exceeded(self):
        """Test that retry raises exception when max retries exceeded."""
        call_count = 0

        @retry(max_retries=2, initial_delay=0.01)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with self.assertRaises(Exception) as cm:
            always_failing_function()

        self.assertIn("Maximum number of retries (2) exceeded", str(cm.exception))
        self.assertEqual(call_count, 3)  # Initial call + 2 retries

    def test_retry_specific_errors_only(self):
        """Test that retry only catches specified error types."""

        @retry(errors=(ConnectionError,), max_retries=2, initial_delay=0.01)
        def function_with_value_error():
            raise ValueError("This should not be retried")

        with self.assertRaises(ValueError):
            function_with_value_error()

    def test_retry_with_custom_parameters(self):
        """Test retry with custom exponential base and jitter."""
        call_count = 0

        @retry(max_retries=2, initial_delay=0.1, exponential_base=3, jitter=False)
        def function_for_delay_test():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Fail")
            return "success"

        with patch("time.sleep") as mock_sleep:
            result = function_for_delay_test()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
        # Check that sleep was called with exponentially increasing delays
        self.assertEqual(len(mock_sleep.call_args_list), 2)

    def test_retry_decorator_without_parentheses(self):
        """Test that retry can be used without parentheses."""
        call_count = 0

        @retry
        def simple_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Fail once")
            return "success"

        result = simple_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)


class TestAsyncRetryDecorator(unittest.TestCase):
    """Test cases for the asynchronous retry decorator."""

    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()

    def test_async_retry_success_on_first_attempt(self):
        """Test that async_retry works when function succeeds on first attempt."""
        call_count = 0

        @async_retry(max_retries=3)
        async def successful_async_function():
            nonlocal call_count
            call_count += 1
            return "async_success"

        async def run_test():
            result = await successful_async_function()
            self.assertEqual(result, "async_success")
            self.assertEqual(call_count, 1)

        self.loop.run_until_complete(run_test())

    def test_async_retry_success_after_failures(self):
        """Test that async_retry works when function succeeds after some failures."""
        call_count = 0

        @async_retry(max_retries=3, initial_delay=0.01)
        async def eventually_successful_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Async connection failed")
            return "async_success"

        async def run_test():
            result = await eventually_successful_async_function()
            self.assertEqual(result, "async_success")
            self.assertEqual(call_count, 3)

        self.loop.run_until_complete(run_test())

    def test_async_retry_max_retries_exceeded(self):
        """Test that async_retry raises exception when max retries exceeded."""
        call_count = 0

        @async_retry(max_retries=2, initial_delay=0.01)
        async def always_failing_async_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails async")

        async def run_test():
            with self.assertRaises(Exception) as cm:
                await always_failing_async_function()

            self.assertIn("Maximum number of retries (2) exceeded", str(cm.exception))
            self.assertEqual(call_count, 3)  # Initial call + 2 retries

        self.loop.run_until_complete(run_test())

    def test_async_retry_specific_errors_only(self):
        """Test that async_retry only catches specified error types."""

        @async_retry(errors=(ConnectionError,), max_retries=2, initial_delay=0.01)
        async def async_function_with_value_error():
            raise ValueError("This should not be retried in async")

        async def run_test():
            with self.assertRaises(ValueError):
                await async_function_with_value_error()

        self.loop.run_until_complete(run_test())

    def test_async_retry_uses_asyncio_sleep(self):
        """Test that async_retry uses asyncio.sleep instead of time.sleep."""
        call_count = 0

        @async_retry(max_retries=2, initial_delay=0.1)
        async def async_function_for_sleep_test():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Fail")
            return "success"

        async def run_test():
            with patch("asyncio.sleep") as mock_async_sleep:
                mock_async_sleep.return_value = asyncio.Future()
                mock_async_sleep.return_value.set_result(None)

                result = await async_function_for_sleep_test()

            self.assertEqual(result, "success")
            self.assertEqual(call_count, 3)
            # Check that asyncio.sleep was called
            self.assertEqual(len(mock_async_sleep.call_args_list), 2)

        self.loop.run_until_complete(run_test())

    def test_async_retry_decorator_without_parentheses(self):
        """Test that async_retry can be used without parentheses."""
        call_count = 0

        @async_retry
        async def simple_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Fail once async")
            return "async_success"

        async def run_test():
            result = await simple_async_function()
            self.assertEqual(result, "async_success")
            self.assertEqual(call_count, 2)

        self.loop.run_until_complete(run_test())

    def test_async_retry_preserves_function_metadata(self):
        """Test that async_retry preserves function name and docstring."""

        @async_retry(max_retries=1)
        async def documented_function():
            """This is a test function."""
            return "test"

        self.assertEqual(documented_function.__name__, "documented_function")
        self.assertEqual(documented_function.__doc__, "This is a test function.")


class TestRetryIntegration(unittest.TestCase):
    """Integration tests for retry decorators."""

    def test_both_decorators_coexist(self):
        """Test that both sync and async retry decorators can be used together."""
        sync_result = None
        async_result = None

        @retry(max_retries=1, initial_delay=0.01)
        def sync_func():
            return "sync_works"

        @async_retry(max_retries=1, initial_delay=0.01)
        async def async_func():
            return "async_works"

        # Test sync
        sync_result = sync_func()

        # Test async
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_result = loop.run_until_complete(async_func())
        finally:
            loop.close()

        self.assertEqual(sync_result, "sync_works")
        self.assertEqual(async_result, "async_works")


if __name__ == "__main__":
    unittest.main()
