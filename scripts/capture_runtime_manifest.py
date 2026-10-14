"""Capture local runtime versions and model identity without loading weights."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from src import config


def capture_manifest() -> dict:
    packages = {}
    for name in [
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "sentence-transformers",
        "chromadb",
        "textual",
        "httpx",
    ]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "model": config.OLLAMA_MODEL,
        "embedding": config.EMBED_MODEL,
        "embedding_revision": config.EMBED_REVISION,
        "torch_contract": {
            "torch_version": packages.get("torch"),
            "torchvision_version": packages.get("torchvision"),
            "cuda_flavor": (packages.get("torch") or "").partition("+")[2] or None,
        },
        "devices": {
            "ingest": config.INGEST_EMBED_DEVICE,
            "query": config.QUERY_EMBED_DEVICE,
        },
        "context_tokens": config.OLLAMA_CONTEXT_TOKENS,
        "output_tokens": config.OLLAMA_OUTPUT_TOKENS,
    }
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        result["hardware"] = gpu.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["hardware_error"] = str(exc)
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            v = client.get(config.OLLAMA_URL + "/api/version")
            v.raise_for_status()
            result["ollama_version"] = v.json()["version"]
            tags = client.get(config.OLLAMA_URL + "/api/tags")
            tags.raise_for_status()
            installed = tags.json().get("models", [])
            result["installed_models"] = [
                {"name": model.get("name"), "digest": model.get("digest"), "details": model.get("details", {})}
                for model in installed
            ]
            match = next((m for m in installed if m.get("name") == config.OLLAMA_MODEL), None)
            if match is None:
                result["model_error"] = "Configured model is not installed"
            else:
                result["digest"] = match["digest"]
                result["model_details"] = match.get("details", {})
                result["quantization"] = result["model_details"].get(
                    "quantization_level"
                )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        result["ollama_error"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=config.RUNTIME_MANIFEST_PATH)
    args = parser.parse_args()
    report = capture_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 1 if "ollama_error" in report or "model_error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
