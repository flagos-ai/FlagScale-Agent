"""Tests for retry_with_backoff."""

import pytest

from flagscale_agent.react.retry import (
    retry_with_backoff, _is_retryable_exception, _is_context_limit_error,
)


class FakeAPIError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"API error {status_code}")


class ConnectionError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class TestRetryWithBackoff:
    def test_success_no_retry(self):
        calls = []
        def fn():
            calls.append(1)
            return "ok"
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(calls) == 1

    def test_retry_on_429(self):
        attempts = []
        def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise FakeAPIError(429)
            return "ok"
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(attempts) == 3

    def test_no_retry_on_400(self):
        def fn():
            raise FakeAPIError(400)
        with pytest.raises(FakeAPIError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)

    def test_exhausted_retries(self):
        def fn():
            raise FakeAPIError(500)
        with pytest.raises(FakeAPIError):
            retry_with_backoff(fn, max_retries=2, base_delay=0.01)

    def test_non_api_error_no_retry(self):
        def fn():
            raise ValueError("bad input")
        with pytest.raises(ValueError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)

    def test_retry_on_connection_error(self):
        attempts = []
        def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("connection reset")
            return "ok"
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(attempts) == 3

    def test_retry_on_api_connection_error(self):
        attempts = []
        def fn():
            attempts.append(1)
            if len(attempts) < 2:
                raise APIConnectionError("failed to connect")
            return "ok"
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(attempts) == 2

    def test_retry_on_api_timeout_error(self):
        attempts = []
        def fn():
            attempts.append(1)
            if len(attempts) < 2:
                raise APITimeoutError("timed out")
            return "ok"
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert len(attempts) == 2

    def test_connection_error_exhausted(self):
        def fn():
            raise ConnectionError("always fails")
        with pytest.raises(ConnectionError):
            retry_with_backoff(fn, max_retries=2, base_delay=0.01)


class TestIsRetryableException:
    def test_connection_error(self):
        assert _is_retryable_exception(ConnectionError("test"))

    def test_api_connection_error(self):
        assert _is_retryable_exception(APIConnectionError("test"))

    def test_api_timeout_error(self):
        assert _is_retryable_exception(APITimeoutError("test"))

    def test_value_error_not_retryable(self):
        assert not _is_retryable_exception(ValueError("test"))

    def test_generic_exception_not_retryable(self):
        assert not _is_retryable_exception(Exception("test"))


class TestIsContextLimitError:
    def test_context_length(self):
        assert _is_context_limit_error(Exception("context length exceeded"))

    def test_prompt_too_long(self):
        assert _is_context_limit_error(Exception("prompt is too long for this model"))

    def test_request_too_large(self):
        assert _is_context_limit_error(Exception("Request too large"))

    def test_too_many_tokens(self):
        assert _is_context_limit_error(Exception("too many tokens in the input"))

    def test_unrelated_400(self):
        assert not _is_context_limit_error(Exception("invalid parameter: temperature"))

    def test_empty_message(self):
        assert not _is_context_limit_error(Exception(""))


class TestContextOverflowRecovery:
    def test_400_context_limit_triggers_callback(self):
        attempts = []
        callback_called = []

        def fn():
            attempts.append(1)
            if len(attempts) == 1:
                e = FakeAPIError(400)
                e.args = ("prompt is too long",)
                raise e
            return "ok"

        def on_overflow():
            callback_called.append(1)
            return True

        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01,
                                     on_context_overflow=on_overflow)
        assert result == "ok"
        assert len(callback_called) == 1
        assert len(attempts) == 2

    def test_400_context_limit_callback_only_once(self):
        attempts = []
        callback_called = []

        class ContextError(Exception):
            def __init__(self):
                self.status_code = 400
                super().__init__("context length exceeded")

        def fn():
            attempts.append(1)
            raise ContextError()

        def on_overflow():
            callback_called.append(1)
            return True

        with pytest.raises(ContextError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01,
                               on_context_overflow=on_overflow)
        assert len(callback_called) == 1

    def test_400_non_context_error_no_callback(self):
        callback_called = []

        def fn():
            raise FakeAPIError(400)

        def on_overflow():
            callback_called.append(1)
            return True

        with pytest.raises(FakeAPIError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01,
                               on_context_overflow=on_overflow)
        assert len(callback_called) == 0

    def test_no_callback_provided(self):
        class ContextError(Exception):
            def __init__(self):
                self.status_code = 400
                super().__init__("context length exceeded")

        def fn():
            raise ContextError()

        with pytest.raises(ContextError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)

    def test_callback_returns_false_raises(self):
        class ContextError(Exception):
            def __init__(self):
                self.status_code = 400
                super().__init__("too many tokens")

        def fn():
            raise ContextError()

        def on_overflow():
            return False

        with pytest.raises(ContextError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01,
                               on_context_overflow=on_overflow)
