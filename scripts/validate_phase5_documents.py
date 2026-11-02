"""Exclusive host evidence for restricted Typst; human acceptance stays pending."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.documents import MATH_EXAMPLES
from src.typst_compile import compile_typst, compiler_capability


async def validate_documents(*, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        available, reason = await asyncio.to_thread(compiler_capability)
        report = {
            "accepted": False,
            "human_review": None,
            "compiler": {"available": available, "reason": reason},
            "cases": [],
            "code_sha256": {
                name: hashlib.sha256((config.ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "src/config.py",
                    "src/documents.py",
                    "src/typst_compile.py",
                )
            },
            "unit_contracts": "Run tests/test_documents.py; fake retry results are not host evidence.",
        }
        if available:
            host = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_typst_compile.py",
                    "tests/test_documents.py::test_host_worked_math_examples_compile_and_open",
                ],
                cwd=config.ROOT,
                env={**os.environ, "AXIOM_RUN_TYPST_TESTS": "1"},
                capture_output=True,
                text=True,
                timeout=6
                * (config.TYPST_COMPILE_SECONDS + config.TYPST_CLEANUP_SECONDS),
                check=False,
            )
            report["host_security_tests"] = {
                "returncode": host.returncode,
                "stdout": host.stdout,
                "stderr": host.stderr,
            }
            binary = Path(shutil.which(config.TYPST_BINARY))
            report["compiler"].update(
                sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                version=(
                    await asyncio.to_thread(
                        subprocess.check_output,
                        [str(binary), "--version"],
                        timeout=config.TYPST_CLEANUP_SECONDS,
                    )
                )
                .decode()
                .strip(),
            )
            cases = [(f"math-{i}", text) for i, text in enumerate(MATH_EXAMPLES, 1)]
            cases += [
                ("host-read", '#read("/etc/passwd")'),
                ("package-fetch", '#import "@preview/cetz:0.3.0": *'),
            ]
            for name, text in cases:
                with tempfile.TemporaryDirectory(prefix="axiom-typst-") as folder:
                    job = Path(folder)
                    (job / "document.typ").write_text(text)
                    start = time.monotonic()
                    result = await compile_typst(
                        job, remaining=config.TYPST_COMPILE_SECONDS
                    )
                    row = {
                        "case": name,
                        "status": result.status,
                        "diagnostics": result.diagnostics,
                        "seconds": time.monotonic() - start,
                        "human_review": None,
                    }
                    if result.pdf_path:
                        destination = output.parent / f"{name}.pdf"
                        with destination.open("xb") as pdf:
                            pdf.write(result.pdf_path.read_bytes())
                        row["pdf_path"] = str(destination.resolve())
                    report["cases"].append(row)
        else:
            report["blocked_gates"] = [
                "real math compilation/PDF opening",
                "compiler resource and process-tree isolation",
                "live generation and content fidelity",
            ]
        statuses = {case["case"]: case["status"] for case in report["cases"]}
        report["checks_passed"] = (
            available
            and report.get("host_security_tests", {}).get("returncode") == 0
            and all(statuses.get(f"math-{i}") == "ok" for i in (1, 2, 3))
            and all(
                statuses.get(name) == "syntax_error"
                for name in ("host-read", "package-fetch")
            )
        )
        json.dump(report, handle, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(validate_documents(output=args.output))
    print(
        json.dumps(
            {
                "compiler": report["compiler"],
                "checks_passed": report["checks_passed"],
                "report": str(args.output.resolve()),
            }
        )
    )
    if not report["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
