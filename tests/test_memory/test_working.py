"""Tests for working memory."""

from src.memory.working import WorkingMemory


def test_working_memory_defaults():
    wm = WorkingMemory()
    assert wm.regime == "unknown"
    assert wm.interpreted_tickers == []
    assert wm.vetoed_tickers == []
    assert wm.approved_tickers == []
    assert wm.total_cost_usd == 0.0
    assert len(wm.run_id) == 12


def test_record_interpretation():
    wm = WorkingMemory()
    wm.record_interpretation("AAPL")
    wm.record_interpretation("MSFT")
    wm.record_interpretation("AAPL")  # duplicate
    assert wm.interpreted_tickers == ["AAPL", "MSFT"]


def test_record_veto():
    wm = WorkingMemory()
    wm.record_veto("BAD")
    wm.record_veto("BAD")  # duplicate
    assert wm.vetoed_tickers == ["BAD"]


def test_record_approval():
    wm = WorkingMemory()
    wm.record_approval("GOOD")
    assert wm.approved_tickers == ["GOOD"]


def test_add_note():
    wm = WorkingMemory()
    wm.add_note("AAPL has high correlation with MSFT")
    assert len(wm.agent_notes) == 1


def test_add_cost():
    wm = WorkingMemory()
    wm.add_cost(100, 0.01)
    wm.add_cost(200, 0.02)
    assert wm.total_tokens == 300
    assert wm.total_cost_usd == 0.03


def test_to_prompt_context():
    wm = WorkingMemory(regime="bull")
    wm.record_interpretation("AAPL")
    wm.record_veto("BAD")
    wm.record_approval("GOOD")
    wm.add_cost(500, 0.05)

    ctx = wm.to_prompt_context()
    assert "bull" in ctx
    assert "AAPL" in ctx
    assert "BAD" in ctx
    assert "GOOD" in ctx
    assert "$0.0500" in ctx


def test_to_prompt_context_empty():
    wm = WorkingMemory()
    ctx = wm.to_prompt_context()
    assert "Run ID" in ctx
    assert "unknown" in ctx


def test_record_verifier_feedback():
    wm = WorkingMemory()
    wm.record_verifier_feedback(["improve thesis", "add risk flags"])
    assert wm.verifier_feedback == ["improve thesis", "add risk flags"]
    wm.record_verifier_feedback(["check sector exposure"])
    assert len(wm.verifier_feedback) == 3


def test_verifier_feedback_in_prompt_context():
    wm = WorkingMemory()
    wm.record_verifier_feedback(["strengthen bull thesis for AAPL"])
    ctx = wm.to_prompt_context()
    assert "strengthen bull thesis for AAPL" in ctx
    assert "verifier feedback" in ctx.lower()


def test_convergence_state():
    wm = WorkingMemory()
    assert wm.convergence_state == "running"
    wm.set_convergence("converged")
    assert wm.convergence_state == "converged"
    wm.set_convergence("budget_exhausted")
    assert wm.convergence_state == "budget_exhausted"
