import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from app.engine import Engine, new_state
from app.llm import MockLLM
from app.models import Autonomy

SAMPLES = Path(__file__).parent.parent / "app" / "samples"


def run_paper(name: str, level: Autonomy, answer=lambda pending: pending["options"][0], llm=None):
    """Drive the step engine the way the browser does, answering every pause with `answer`."""
    async def go():
        engine = Engine(llm or MockLLM(delay=0))
        st = new_state((SAMPLES / name).read_text(), level, "demo", engine.llm.name)
        events, reply, steps = [], None, 0
        while True:
            st, ev = await engine.advance(st, reply)
            events += ev
            steps += 1
            reply = answer(st["pending"]) if st["status"] == "waiting" else None
            if st["status"] not in ("running", "waiting") or steps > 50:
                break
        return SimpleNamespace(status=st["status"], events=events, steps=steps,
                               report=st["report"] if st["status"] == "done" else None)
    return asyncio.run(go())


def types(run):
    return [e["type"] for e in run.events]


def test_injection_paper_pauses_at_balanced_and_notes_integrity():
    run = run_paper("federated-gait.md", Autonomy.BALANCED)
    assert run.status == "done"
    asks = [e for e in run.events if e["type"] == "ask"]
    assert asks[0]["checkpoint"] == "input"
    assert "integrity_note" in run.report


def test_evaluator_retry_is_visible():
    run = run_paper("sparse-routing.md", Autonomy.HANDS_OFF)
    evals = [e for e in run.events if e["type"] == "eval" and e["stage"] == "critic"]
    assert [e["passed"] for e in evals] == [False, True]


def test_hands_off_never_pauses_on_clean_paper():
    run = run_paper("sparse-routing.md", Autonomy.HANDS_OFF)
    assert "ask" not in types(run) and run.status == "done"


def test_hands_on_asks_more_than_hands_off():
    on = run_paper("sparse-routing.md", Autonomy.HANDS_ON)
    off = run_paper("sparse-routing.md", Autonomy.HANDS_OFF)
    assert types(on).count("ask") > types(off).count("ask")


def test_human_can_override_the_verdict():
    def pick(p):
        return "Change to minor revision" if p["checkpoint"] == "verdict" else p["options"][0]
    run = run_paper("sparse-routing.md", Autonomy.HANDS_ON, answer=pick)
    assert run.report["recommendation"] == "minor revision" and run.report["human_override"]


def test_stop_at_input_checkpoint():
    run = run_paper("federated-gait.md", Autonomy.HANDS_ON, answer=lambda p: "Stop the review")
    assert run.status == "stopped" and run.report is None


class AlwaysUngrounded(MockLLM):
    async def generate(self, agent, system, user, ctx):
        data = json.loads(await super().generate(agent, system, user, ctx))
        for c in data.get("claims", []):
            c["evidence"] = [{"chunk_id": "c0", "quote": "made-up text that is not in the paper"}]
        return json.dumps(data)


def test_stops_after_three_failed_attempts_instead_of_guessing():
    run = run_paper("sparse-routing.md", Autonomy.HANDS_OFF, llm=AlwaysUngrounded(delay=0))
    fails = [e for e in run.events if e["type"] == "eval" and not e["passed"]]
    assert len(fails) == 3 and run.status == "stopped"


class RejectedKey(MockLLM):
    async def generate(self, agent, system, user, ctx):
        import httpx
        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        raise httpx.HTTPStatusError("401", request=req, response=httpx.Response(401, request=req))


def test_bad_api_key_gives_a_clear_error_without_retrying():
    run = run_paper("sparse-routing.md", Autonomy.HANDS_OFF, llm=RejectedKey(delay=0))
    assert run.status == "error"
    assert "API key was rejected" in run.events[-1]["message"]
    assert not [e for e in run.events if e["type"] == "eval"]


def test_each_step_makes_at_most_one_model_call():
    calls = []

    class Counting(MockLLM):
        async def generate(self, agent, system, user, ctx):
            calls.append(agent)
            return await super().generate(agent, system, user, ctx)

    async def go():
        engine = Engine(Counting(delay=0))
        st = new_state((SAMPLES / "sparse-routing.md").read_text(), Autonomy.HANDS_OFF, "demo", "m")
        per_step = []
        while st["status"] == "running":
            before = len(calls)
            st, _ = await engine.advance(st)
            per_step.append(len(calls) - before)
        return st, per_step
    st, per_step = asyncio.run(go())
    assert st["status"] == "done" and max(per_step) == 1 and sum(per_step) == len(calls)


def test_state_is_json_serialisable_between_steps():
    async def go():
        engine = Engine(MockLLM(delay=0))
        st = new_state((SAMPLES / "federated-gait.md").read_text(), Autonomy.HANDS_OFF, "demo", "m")
        while st["status"] == "running":
            st = json.loads(json.dumps(st))          # what the browser does
            st, _ = await engine.advance(st)
        return st
    assert asyncio.run(go())["status"] == "done"


def test_rate_limit_pauses_without_losing_progress():
    from app.llm import RateLimited

    class Busy(MockLLM):
        hits = 0

        async def generate(self, agent, system, user, ctx):
            if agent == "critic" and Busy.hits == 0:
                Busy.hits += 1
                raise RateLimited(12)
            return await super().generate(agent, system, user, ctx)

    run = run_paper("sparse-routing.md", Autonomy.HANDS_OFF, llm=Busy(delay=0))
    assert run.status == "done"
    assert any(e["type"] == "wait" and e["seconds"] == 12 for e in run.events)


def test_toxic_review_is_rewritten_once():
    class RudeOnce(MockLLM):
        rude = True

        async def generate(self, agent, system, user, ctx):
            out = json.loads(await super().generate(agent, system, user, ctx))
            if agent == "writer" and RudeOnce.rude:
                RudeOnce.rude = False
                out["summary"] = "Frankly the evaluation is garbage."
            return json.dumps(out)

    run = run_paper("sparse-routing.md", Autonomy.HANDS_OFF, llm=RudeOnce(delay=0))
    guards = [e for e in run.events if e["type"] == "step" and e["stage"] == "output_guardrails" and e["status"] == "done"]
    assert run.status == "done" and guards[0]["findings"] and not guards[-1]["findings"]
    assert "garbage" not in run.report["summary"]
