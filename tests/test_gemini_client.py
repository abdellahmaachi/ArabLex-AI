"""Tests Gemini sans clé réelle, réseau ni facturation."""

from unittest.mock import Mock

import pytest
from google.genai import errors, types

from src.rag import gemini_client
from src.rag.gemini_client import (
    GeminiBlockedError,
    GeminiClient,
    GeminiRequestError,
    GeminiResponseError,
    GeminiTruncatedError,
)


@pytest.fixture
def sdk(monkeypatch):
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(gemini_client.genai, "Client", factory)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-flash")
    return client, factory


def response(reason="STOP", text="نص الجواب [S1]"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason=reason,
        content=types.Content(parts=[types.Part(text=text)]),
    )])


def test_generation_configuration_text_and_usage(sdk):
    client, factory = sdk
    result = response()
    result.candidates[0].content.parts.insert(0, types.Part(text="private thought", thought=True))
    result.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=12, candidates_token_count=7, total_token_count=22,
    )
    result.model_version = "test-flash-version"
    client.models.generate_content.return_value = result
    with GeminiClient(timeout_ms=5000, max_attempts=2) as wrapper:
        answer = wrapper.generate("  سؤال\n")
        assert answer.text == "نص الجواب [S1]"
        assert answer.model == "test-flash-version"
        assert (answer.prompt_tokens, answer.output_tokens, answer.total_tokens) == (12, 7, 22)
        assert answer.finish_reason == "STOP"
        call = client.models.generate_content.call_args.kwargs
        assert call["contents"] == "  سؤال\n"
        assert call["model"] == "test-flash"
        assert call["config"].candidate_count == 1
        assert wrapper.generate_text("سؤال") == answer.text
    options = factory.call_args.kwargs["http_options"]
    assert options.timeout == 5000
    assert options.retry_options.attempts == 2
    assert options.retry_options.http_status_codes == [408, 429, 500, 502, 503, 504]
    assert factory.call_args.kwargs["vertexai"] is False
    wrapper.close()
    client.close.assert_called_once()
    with pytest.raises(RuntimeError, match="fermé"):
        wrapper.generate("سؤال")


@pytest.mark.parametrize("code,retryable", [(400, False), (401, False), (403, False),
                                          (404, False), (429, True), (503, True)])
def test_api_error_mapping_without_sensitive_details(sdk, code, retryable):
    client, _ = sdk
    client.models.generate_content.side_effect = errors.APIError(
        code, {"error": {"message": "test-key secret prompt"}},
    )
    with GeminiClient() as wrapper:
        with pytest.raises(GeminiRequestError) as caught:
            wrapper.generate("secret prompt")
    assert caught.value.status_code == code
    assert caught.value.retryable is retryable
    assert "secret" not in str(caught.value)
    assert "test-key" not in str(caught.value)
    # Les tentatives sont déléguées au SDK, pas doublées par le wrapper.
    assert client.models.generate_content.call_count == 1


def test_transport_error(sdk):
    client, _ = sdk
    client.models.generate_content.side_effect = gemini_client._TRANSPORT_ERRORS[0]("private URL")
    with GeminiClient() as wrapper:
        with pytest.raises(GeminiRequestError) as caught:
            wrapper.generate("سؤال")
    assert caught.value.retryable
    assert caught.value.status_code is None
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("reason,error", [("MAX_TOKENS", GeminiTruncatedError),
    ("SAFETY", GeminiBlockedError), ("RECITATION", GeminiBlockedError),
    ("OTHER", GeminiResponseError), (None, GeminiResponseError)])
def test_incomplete_responses_are_not_successes(sdk, reason, error):
    client, _ = sdk
    client.models.generate_content.return_value = response(reason=reason)
    with GeminiClient() as wrapper:
        with pytest.raises(error) as caught:
            wrapper.generate("سؤال")
    assert caught.value.reason == reason


def test_blocked_prompt(sdk):
    client, _ = sdk
    client.models.generate_content.return_value = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"),
    )
    with GeminiClient() as wrapper:
        with pytest.raises(GeminiBlockedError):
            wrapper.generate("سؤال")


@pytest.mark.parametrize("result", [None, types.GenerateContentResponse(), response(text=" \n")])
def test_empty_responses(sdk, result):
    client, _ = sdk
    client.models.generate_content.return_value = result
    with GeminiClient() as wrapper:
        with pytest.raises(GeminiResponseError):
            wrapper.generate("سؤال")


def test_invalid_prompts_never_call_sdk(sdk):
    client, _ = sdk
    with GeminiClient() as wrapper:
        for prompt, error in [(None, TypeError), ("", ValueError), (" \n", ValueError)]:
            with pytest.raises(error):
                wrapper.generate(prompt)
    client.models.generate_content.assert_not_called()


@pytest.mark.parametrize("kwargs", [
    {"api_key": ""}, {"model": " "}, {"timeout_ms": 0}, {"max_attempts": True},
    {"max_output_tokens": -1}, {"temperature": float("nan")}, {"temperature": 3},
])
def test_invalid_configuration(sdk, kwargs):
    _, factory = sdk
    with pytest.raises(ValueError):
        GeminiClient(**kwargs)
    factory.assert_not_called()


def test_missing_environment_and_explicit_override(sdk, monkeypatch):
    _, factory = sdk
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient()
    monkeypatch.delenv("GEMINI_MODEL")
    with pytest.raises(ValueError, match="GEMINI_MODEL"):
        GeminiClient(api_key="explicit")
    with GeminiClient(api_key="explicit", model="explicit-flash"):
        assert factory.call_args.kwargs["api_key"] == "explicit"
