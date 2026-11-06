from app.autonomy import claims_checkpoint, input_checkpoint, verdict_checkpoint
from app.evaluator import evaluate
from app.ingest import parse
from app.models import Autonomy

PAPER = parse("""# A Paper
## Abstract
We propose a method that improves accuracy by ten points on a benchmark of reading tasks.
## Results
The method reaches 81.2 accuracy while the baseline reaches 71.0 on the same split.
""")


def test_grounded_quote_passes():
    data = {"title": "A Paper", "summary": "s", "claims": [{
        "text": "improves accuracy", "confidence": 0.9,
        "evidence": [{"chunk_id": "c1", "quote": "The method reaches 81.2 accuracy"}]}]}
    _, problems = evaluate("extractor", data, PAPER)
    assert problems == []


def test_paraphrased_quote_is_rejected():
    data = {"title": "A Paper", "summary": "s", "claims": [{
        "text": "improves accuracy", "confidence": 0.9,
        "evidence": [{"chunk_id": "c1", "quote": "The approach gets about 81 accuracy"}]}]}
    _, problems = evaluate("extractor", data, PAPER)
    assert any("not verbatim" in p for p in problems)


def test_unknown_chunk_and_bad_schema_are_rejected():
    _, p1 = evaluate("extractor", {"title": "t", "summary": "s", "claims": [{
        "text": "x", "confidence": 0.5, "evidence": [{"chunk_id": "c99", "quote": "x"}]}]}, PAPER)
    assert any("does not exist" in p for p in p1)
    _, p2 = evaluate("extractor", {"title": "t"}, PAPER)
    assert p2 and p2[0].startswith("schema")


def test_writer_consistency_rules():
    issue = {"aspect": "baselines", "severity": "major", "comment": "c", "confidence": 0.8,
             "evidence": [{"chunk_id": "c1", "quote": "the baseline reaches 71.0"}]}
    data = {"summary": "s", "strengths": ["x"], "weaknesses": [issue, {**issue, "aspect": "stats"}],
            "questions_for_authors": ["q"], "scores": {"soundness": 5, "clarity": 4, "contribution": 4},
            "recommendation": "accept", "confidence": 0.7}
    _, problems = evaluate("writer", data, PAPER)
    assert any("accept" in p for p in problems)
    assert any("soundness" in p for p in problems)


def test_autonomy_threshold_changes_behaviour():
    inj = [{"type": "prompt_injection", "detail": "ignore previous instructions"}]
    assert input_checkpoint(Autonomy.BALANCED, inj).action == "ask"
    assert input_checkpoint(Autonomy.HANDS_OFF, inj).action == "proceed"
    assert input_checkpoint(Autonomy.HANDS_ON, []).action == "proceed"


def test_risk_grows_with_unsupported_claims_and_retries():
    weak = [{"text": "x", "confidence": 0.3}] * 2
    strong = [{"text": "y", "confidence": 0.9}] * 2
    assert claims_checkpoint(Autonomy.BALANCED, weak, 1).risk > claims_checkpoint(Autonomy.BALANCED, strong, 0).risk
    assert claims_checkpoint(Autonomy.BALANCED, strong, 0).action == "proceed"


def test_reject_verdict_always_pauses_at_balanced():
    d = verdict_checkpoint(Autonomy.BALANCED, "reject", 0.9, 4)
    assert d.action == "ask" and "Finalize" in d.options
