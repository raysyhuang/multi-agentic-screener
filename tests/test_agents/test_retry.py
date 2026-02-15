"""Tests for retry infrastructure."""

import pytest

from src.agents.retry import (
    RetryPolicy,
    RetryResult,
    AttemptRecord,
    FailureReason,
    build_retry_prompt_suffix,
)


def test_retry_policy_should_retry_parse_error():
    policy = RetryPolicy(retry_on_parse_error=True)
    assert policy.should_retry(FailureReason.PARSE_ERROR) is True


def test_retry_policy_no_retry_parse_error():
    policy = RetryPolicy(retry_on_parse_error=False)
    assert policy.should_retry(FailureReason.PARSE_ERROR) is False


def test_retry_policy_never_retries_fundamental_reject():
    policy = RetryPolicy(
        retry_on_parse_error=True,
        retry_on_api_error=True,
        retry_on_low_quality=True,
    )
    assert policy.should_retry(FailureReason.FUNDAMENTAL_REJECT) is False


def test_retry_policy_api_error():
    policy = RetryPolicy(retry_on_api_error=True)
    assert policy.should_retry(FailureReason.LLM_API_ERROR) is True

    policy_no = RetryPolicy(retry_on_api_error=False)
    assert policy_no.should_retry(FailureReason.LLM_API_ERROR) is False


def test_retry_policy_low_quality():
    policy = RetryPolicy(retry_on_low_quality=False)
    assert policy.should_retry(FailureReason.LOW_QUALITY) is False

    policy_yes = RetryPolicy(retry_on_low_quality=True)
    assert policy_yes.should_retry(FailureReason.LOW_QUALITY) is True


def test_retry_result_tracks_attempts():
    result = RetryResult()
    assert result.attempt_count == 0
    assert not result.succeeded

    result.add_attempt(AttemptRecord(attempt_num=1, success=False, failure_reason=FailureReason.PARSE_ERROR))
    assert result.attempt_count == 1
    assert not result.succeeded

    result.add_attempt(AttemptRecord(attempt_num=2, success=True))
    result.value = "some_value"
    assert result.attempt_count == 2
    assert result.succeeded


def test_retry_result_failure_reasons():
    result = RetryResult()
    result.add_attempt(AttemptRecord(
        attempt_num=1, success=False, failure_reason=FailureReason.PARSE_ERROR
    ))
    result.add_attempt(AttemptRecord(
        attempt_num=2, success=False, failure_reason=FailureReason.LLM_API_ERROR
    ))
    assert result.failure_reasons == [FailureReason.PARSE_ERROR, FailureReason.LLM_API_ERROR]


def test_retry_result_cost_tracking():
    result = RetryResult()
    result.add_cost(100, 0.01)
    result.add_cost(200, 0.02)
    assert result.total_tokens == 300
    assert result.total_cost_usd == pytest.approx(0.03)


def test_build_retry_prompt_suffix_parse_error():
    suffix = build_retry_prompt_suffix(2, FailureReason.PARSE_ERROR, "invalid json")
    assert "RETRY" in suffix
    assert "Attempt 2" in suffix
    assert "JSON" in suffix
    assert "invalid json" in suffix


def test_build_retry_prompt_suffix_api_error():
    suffix = build_retry_prompt_suffix(2, FailureReason.LLM_API_ERROR)
    assert "RETRY" in suffix
    assert "API error" in suffix


def test_build_retry_prompt_suffix_low_quality():
    suffix = build_retry_prompt_suffix(2, FailureReason.LOW_QUALITY, "thesis too short")
    assert "RETRY" in suffix
    assert "quality" in suffix.lower()
    assert "thesis too short" in suffix


def test_attempt_record_fields():
    record = AttemptRecord(
        attempt_num=1,
        success=False,
        failure_reason=FailureReason.PARSE_ERROR,
        error_message="bad json",
        raw_output='{"broken": ',
    )
    assert record.attempt_num == 1
    assert not record.success
    assert record.failure_reason == FailureReason.PARSE_ERROR
    assert record.error_message == "bad json"
    assert record.raw_output == '{"broken": '
