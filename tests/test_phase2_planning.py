import json

import httpx
import pytest

from scripts import probe_phase2_planning as planning
from src import config


def test_fts5_keeps_stable_identity_and_ranks_actual_text():
    r = planning.check_fts(
        [(1, "interpreter compiler language"), (2, "bread and water")]
    )
    assert r["available"] and r["document_count"] == 2
    assert [row[0] for row in r["sample_ranked_ids"]] == ["page-0001"]


def test_atomic_manifest_primitives_leave_old_until_activation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "VECTOR_STORE", tmp_path)
    r = planning.check_atomic_manifest()
    assert (
        r["old_until_replace"]
        and r["new_after_replace"]
        and r["same_filesystem_as_store"]
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [False, True])
def test_rewrite_is_bounded_local_and_unloads_on_failure(monkeypatch, failure):
    seen = []
    unloaded = []
    real_http = httpx.Client

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        if failure:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(
            200,
            json={
                "response": "Give another interpreter example.",
                "done_reason": "stop",
                "eval_count": 6,
            },
        )

    monkeypatch.setattr(
        planning.httpx,
        "Client",
        lambda **kwargs: real_http(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(
        planning.OllamaClient, "unload_and_wait", lambda _: unloaded.append(True)
    )
    r = planning.rewrite_check(
        {
            "question": "Another example?",
            "history": [
                {"role": "user", "content": "What is an interpreter?"},
                {"role": "assistant", "content": "It evaluates programs."},
            ],
        }
    )
    assert seen[0]["keep_alive"] == 0 and seen[0]["think"] is False
    assert seen[0]["options"]["num_predict"] == config.QUERY_REWRITE_MAX_OUTPUT_TOKENS
    assert unloaded == [True]
    if failure:
        assert r["within_budget"] is False and "offline" in r["error"]
    else:
        assert r["rewritten"] == "Give another interpreter example."
