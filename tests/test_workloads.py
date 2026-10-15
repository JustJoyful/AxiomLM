from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from threading import Barrier, Event
import time

import pytest

from src.workloads import WorkloadCoordinator


def test_generation_and_ingest_are_mutually_exclusive():
    coordinator = WorkloadCoordinator()
    generation = coordinator.try_acquire("generation")
    assert generation
    assert coordinator.try_acquire("ingest") is None
    coordinator.release(generation)
    ingest = coordinator.try_acquire("ingest")
    assert ingest
    assert coordinator.try_acquire("generation") is None


def test_only_owner_can_release_admission():
    coordinator = WorkloadCoordinator()
    owner = coordinator.try_acquire("generation")
    with pytest.raises(ValueError, match="does not own"):
        coordinator.release("not-the-owner")
    assert coordinator.try_acquire("ingest") is None
    coordinator.release(owner)


def test_concurrent_admission_has_exactly_one_owner():
    coordinator = WorkloadCoordinator()
    barrier = Barrier(2)

    def acquire(kind):
        barrier.wait()
        return coordinator.try_acquire(kind)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, ["generation", "ingest"]))
    assert sum(result is not None for result in results) == 1
    coordinator.release(next(result for result in results if result is not None))


def test_tui_uses_one_admission_gate_for_generation_and_ingest():
    from src.tui import AxiomLMApp

    app = AxiomLMApp()
    generation = app._try_acquire_workload("generation")
    assert generation
    assert app._try_acquire_workload("ingest") is None
    app._workload_coordinator.release(generation)
    ingest = app._try_acquire_workload("ingest")
    assert ingest
    app._workload_coordinator.release(ingest)


def test_generation_ownership_is_held_until_worker_actually_returns(monkeypatch):
    from src import tui

    app = __import__("src.tui", fromlist=["AxiomLMApp"]).AxiomLMApp()
    monkeypatch.setattr(
        tui, "OllamaClient", lambda: type("Client", (), {"unload_and_wait": lambda self: None})()
    )
    entered = Event()
    finish = Event()

    def execute(*args):
        entered.set()
        finish.wait(timeout=2)

    app.execute_rag_query = execute
    token = app._try_acquire_workload("generation")
    assert token
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            app._execute_rag_query_with_admission,
            "q",
            "",
            "book",
            "model",
            "request",
            token,
        )
        assert entered.wait(timeout=1)
        assert app._try_acquire_workload("ingest") is None
        finish.set()
        future.result(timeout=1)
    assert app._workload_coordinator.active_kind is None


def test_generation_admission_waits_for_confirmed_ollama_cleanup(monkeypatch):
    from src import tui
    from src.tui import AxiomLMApp

    app = AxiomLMApp()
    events = []
    cleanup_entered = Event()
    finish_cleanup = Event()

    class Client:
        def unload_and_wait(self):
            cleanup_entered.set()
            assert finish_cleanup.wait(timeout=2)
            events.append("unloaded")

    monkeypatch.setattr(tui, "OllamaClient", Client)
    monkeypatch.setattr(app, "execute_rag_query", lambda *_args: events.append("returned"))
    token = app._try_acquire_workload("generation")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            app._execute_rag_query_with_admission, "q", "", "b", "m", "r", token
        )
        assert cleanup_entered.wait(timeout=1)
        assert app._workload_coordinator.active_kind == "generation"
        assert app._try_acquire_workload("ingest") is None
        finish_cleanup.set()
        future.result(timeout=1)
    assert events == ["returned", "unloaded"]
    assert app._workload_coordinator.active_kind is None


def test_unconfirmed_generation_cleanup_keeps_admission_closed(monkeypatch):
    from src import tui
    from src.tui import AxiomLMApp

    app = AxiomLMApp()

    class Client:
        def unload_and_wait(self):
            raise RuntimeError("Ollama still has a model loaded")

    monkeypatch.setattr(tui, "OllamaClient", Client)
    monkeypatch.setattr(app, "execute_rag_query", lambda *_args: None)
    monkeypatch.setattr(app, "call_from_thread", lambda fn, *args: fn(*args))
    token = app._try_acquire_workload("generation")
    app._execute_rag_query_with_admission("q", "", "b", "m", "r", token)
    assert app._workload_coordinator.active_kind == "generation"
    assert app._try_acquire_workload("ingest") is None
    assert app.notices and "cleanup" in app.notices[0].lower()


def test_external_job_setup_failure_terminates_and_untracks_child(tmp_path, monkeypatch):
    from src.tui import AxiomLMApp

    app = AxiomLMApp()
    real_popen = subprocess.Popen
    children = []

    def record_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record_popen)
    with pytest.raises(FileNotFoundError):
        app._run_external_job(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            live_log_path=tmp_path / "missing" / "job.log",
        )
    child = children[0]
    try:
        assert child.poll() is not None
        assert child not in app._active_processes
    finally:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait(timeout=2)


def test_callback_failure_terminates_and_untracks_child(monkeypatch):
    from src.tui import AxiomLMApp

    app = AxiomLMApp()
    real_popen = subprocess.Popen
    children = []

    def record_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record_popen)

    def fail_callback(_line):
        raise RuntimeError("UI callback failed")

    with pytest.raises(RuntimeError, match="UI callback"):
        app._run_external_job(
            [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
            on_output=fail_callback,
        )
    child = children[0]
    try:
        assert child.poll() is not None
        assert child not in app._active_processes
    finally:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait(timeout=2)


def test_external_job_kills_surviving_descendant_after_leader_exits(monkeypatch):
    from src import tui
    from src.tui import AxiomLMApp

    app = AxiomLMApp()
    monkeypatch.setattr(tui, "PROCESS_TERMINATE_GRACE_SECONDS", 0.1)
    script = (
        "import os,signal,time; pid=os.fork(); "
        "(signal.signal(signal.SIGTERM,signal.SIG_IGN),"
        "os.close(1),os.close(2),time.sleep(30),os._exit(0)) if pid==0 "
        "else print(pid,flush=True)"
    )
    code, output = app._run_external_job([sys.executable, "-c", script])
    child_pid = int(output.strip())
    assert code == 0
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            state = open(f"/proc/{child_pid}/stat", encoding="utf-8").read().split()[2]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        time.sleep(0.02)
    else:
        os.kill(child_pid, 9)
        raise AssertionError("Descendant survived after its leader exited.")


def test_shutdown_closes_process_admission_race(monkeypatch):
    from src.tui import AxiomLMApp

    app = AxiomLMApp()
    app._process_shutdown_started = True
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned after process shutdown"),
    )
    with pytest.raises(RuntimeError, match="shutting down"):
        app._run_external_job([sys.executable, "-c", "pass"])


def test_tui_shutdown_kills_descendants_that_ignore_sigterm():
    from src.tui import AxiomLMApp

    app = AxiomLMApp()
    child_code = (
        "import os,signal,time; ready_r,ready_w=os.pipe(); pid=os.fork(); "
        "(os.close(ready_r),signal.signal(signal.SIGTERM,signal.SIG_IGN),"
        "os.write(ready_w,b'ready'),os.close(ready_w),time.sleep(30),os._exit(0)) "
        "if pid==0 else (os.close(ready_w),os.read(ready_r,5),os.close(ready_r),"
        "print(pid,flush=True),time.sleep(30))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = int(process.stdout.readline().strip())
    with app._process_lock:
        app._active_processes.add(process)
    try:
        app._terminate_active_processes()
        process.wait(timeout=2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                state = open(f"/proc/{child_pid}/stat", encoding="utf-8").read().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("SIGTERM-ignoring child remained alive after TUI shutdown.")
    finally:
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass
