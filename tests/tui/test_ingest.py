from src.ui.stream import TurnRecord, TurnView


async def test_ingest_success_links_the_pdf_and_failure_uses_alert(
    app_factory, tmp_path
):
    app = app_factory()
    pdf = tmp_path / "数学 notes.pdf"
    async with app.run_test():
        app._ingest_path = str(pdf)
        app._ingest_view = TurnView(TurnRecord("ingest", "parse", "book", "model"))
        await app.query_one("#chat_scroll").add_turn(app._ingest_view)
        await app._finish_parse_and_index("book", 3, 5, None)
        text = app._ingest_view.step_line.render()
        assert any(span.style.link == pdf.as_uri() for span in text.spans)
        await app._finish_parse_and_index("book", 0, 0, "disk full")
        assert app._ingest_view.step_line.has_class("failed")


async def test_observed_ingest_stages_do_not_invent_percentages(app_factory):
    app = app_factory()
    async with app.run_test():
        app._is_parsing = True
        app._ingest_view = TurnView(
            TurnRecord("ingest", "/parse book.pdf", "book", "model", phase="INGEST")
        )
        await app.query_one("#chat_scroll").add_turn(app._ingest_view)
        app._ingest_output("parse", "Parsing PDF pages")
        assert app._ingest_counts == {}
        app._ingest_output("parse", "pages 2/8")
        assert app._ingest_counts["parse"] == (2, 8)
        app._ingest_output("index", "Chunking by structural headers...")
        assert app._ingest_stage == "chunk"
        app._ingest_output("index", "Adding batch 1...")
        assert app._ingest_stage == "embed"
        assert "embed" not in app._ingest_counts
        app._ingest_output("index", "index 1/4")
        assert app._ingest_counts["index"] == (1, 4)
        app._ingest_cancelled = True
        app._ingest_path = "/tmp/数学 notes.pdf"
        await app._finish_parse_and_index("book", 0, 0, "cancelled")
        assert app._ingest_view.record.phase == "CANCELLED"
        assert "○ cancelled · resume with /parse" in str(
            app._ingest_view.step_line.render()
        )
        assert app.active_collection_name is None


def test_cancel_stops_owned_process_but_preserves_checkpoint(tmp_path):
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from src.tui import AxiomLMApp

    path = tmp_path / "page_0000.md"
    app = AxiomLMApp()
    entered = Event()
    code = "from pathlib import Path; import sys,time; Path(sys.argv[1]).write_text('parsed page'); print('ready',flush=True); time.sleep(30)"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            app._run_external_job,
            [sys.executable, "-c", code, str(path)],
            on_output=lambda _: entered.set(),
        )
        assert entered.wait(2)
        app._ingest_cancelled = True
        app._terminate_job_processes()
        result, _output = future.result(timeout=2)
    assert result != 0
    assert path.read_text() == "parsed page"
    assert not app._active_processes


def test_cancelled_ingest_cannot_launch_the_next_gpu_stage(monkeypatch):
    import pytest

    from src import tui

    app = tui.AxiomLMApp()
    app._ingest_cancelled = True
    monkeypatch.setattr(
        tui.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not start")),
    )
    with pytest.raises(RuntimeError):
        app._run_external_job(["no-new-stage"])


def test_mineru_preserves_explicit_cpu_embedding_choice(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from src import checkpoint, tui

    app = tui.AxiomLMApp()
    app._use_gpu_for_embeddings = False
    app._ingest_path = str(tmp_path / "book.pdf")
    app.call_from_thread = lambda *_args, **_kwargs: None
    app._release_ollama_models = lambda: None
    app._is_mineru_available = lambda: True
    jobs = []
    monkeypatch.setitem(
        sys.modules,
        "src.ocr",
        SimpleNamespace(resolve_mode=lambda *_: ("mineru", "explicit")),
    )
    monkeypatch.setattr(checkpoint, "all_pages", lambda _: [0])
    monkeypatch.setattr(tui, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(
        tui, "get_collection", lambda _: SimpleNamespace(count=lambda: 1)
    )
    monkeypatch.setattr(tui, "ensure_collection_ready", lambda _: None)

    def run(command, **kwargs):
        jobs.append((command, kwargs["env"]))
        return 0, ""

    app._run_external_job = run
    app._parse_and_index_worker(
        app._ingest_path, "mineru", app._try_acquire_workload("ingest")
    )
    assert len(jobs) == 2
    assert "--allow-cpu-embeddings" in jobs[1][0]
    assert jobs[1][1]["AXIOM_EMBED_DEVICE"] == "cpu"
