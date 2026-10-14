import subprocess


def test_setup_dry_run_selects_cuda_profile_without_installing():
    result = subprocess.run(
        ["bash", "scripts/setup_env.sh", "--dry-run", "--no-model"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "uv sync" in result.stdout
    assert "--extra cuda" in result.stdout
    assert "pip install" not in result.stdout
    assert "ollama pull" not in result.stdout
    assert "snapshot_download" not in result.stdout


def test_setup_dry_run_selects_cpu_profile():
    result = subprocess.run(
        ["bash", "scripts/setup_env.sh", "--cpu", "--dry-run", "--no-model"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--extra cpu" in result.stdout
    assert "--extra cuda" not in result.stdout


def test_setup_dry_run_includes_pinned_embedding_and_ollama_models():
    result = subprocess.run(
        ["bash", "scripts/setup_env.sh", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "snapshot_download" in result.stdout
    assert "8a7f00aac812071b69403df470f1038ec85f8925" in result.stdout
    assert "ollama pull qwen3.5:4b-q4_K_M" in result.stdout
