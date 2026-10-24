"""Compile an offline and an online Phase 3 agent report into one review sheet.

The sheet is for a person to judge each answer; it records no verdicts, and
the reports' answer_review blocks stay as the validator wrote them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_phase3_baseline import load_cases
from scripts.validate_phase3_agent import answer_lints, apply_web_expectations
from src import config

REVIEW_BOXES = (
    "**Review:** offline [ ] correct [ ] supported [ ] no invented facts · "
    "online [ ] correct [ ] supported [ ] no invented facts"
)


def _side(title: str, row: dict, case: dict) -> list[str]:
    out = [f"#### {title}", ""]
    a = row.get("assessment") or {}
    out.append(
        f"- Pages retrieved: {row.get('physical_pages') or 'none'} · "
        f"book evidence judged sufficient: {a.get('sufficient')} · "
        f"{row.get('seconds', 0):.1f} s · done: {row.get('done_reason')}"
    )
    rewrite = row.get("rewrite")
    if rewrite:
        if rewrite["applied"]:
            out.append(f"- Retrieval query (rewritten): {rewrite['retrieval_query']}")
        else:
            out.append(f"- Rewrite not applied: {rewrite['fallback_reason']}")
    if a.get("reason"):
        out.append(f"- Assessment reason: {a['reason']}")
    v = row.get("verification")
    if v:
        out.append(f"- Trusted check: **{v['status']}** ({v['reason']})")
    failed = [k for k, ok in row.get("policy", {}).items() if ok is not True]
    out.append(
        f"- Policy gates: {'all pass' if not failed else 'FAIL ' + ', '.join(failed)}"
    )
    lints = row.get("lints")
    if lints is None:
        lints = answer_lints(row, case)
    out.append(f"- Lints (not gates): {'; '.join(lints) if lints else 'none'}")
    if row.get("web_sources"):
        out.append("- Web sources:")
        out += [f"  - <{u}>" for u in row["web_sources"]]
    else:
        out.append("- Web sources: none")
    out += ["", "Answer:", ""]
    out += ["> " + line if line else ">" for line in row.get("answer", "").splitlines()]
    out.append("")
    return out


def _tag(row: dict) -> str:
    gates = all(v is True for v in row.get("policy", {}).values())
    lints = len(row.get("lints") or [])
    return ("pass" if gates else "**fail**") + (f" · {lints} lint" if lints else "")


def build(offline: dict, online: dict, offline_path: Path, online_path: Path) -> str:
    labels, cases = load_cases()
    apply_web_expectations(labels, cases)
    by_id = {c["case_id"]: c for c in cases}
    # load_cases keeps only routing fields; the hash-pinned Phase 2 file has the claims.
    phase2 = json.loads((config.ROOT / labels["phase2_reference"]["path"]).read_text())
    for c in phase2["cases"]:
        if c["case_id"] in by_id:
            by_id[c["case_id"]].setdefault("expected_claims", c.get("expected_claims"))
    rows_on = {r["case_id"]: r for r in online["rows"]}
    sampling = offline.get("sampling")
    note = (
        "Sampling as sent: " + json.dumps(sampling, sort_keys=True) + "."
        if sampling
        else "Sampling options were not recorded in this report."
    )
    lines = [
        (
            f"# Phase 3 answer review: {offline_path.stem} (offline) and "
            f"{online_path.stem} (web)"
        ),
        "",
        f"Offline report: `{offline_path}` (passed={offline['passed']})  ",
        f"Online report: `{online_path}` (passed={online['passed']})  ",
        f"Code: `{offline['git_head']}` at run start. {note}",
        "",
        (
            "For each case, mark the answer as correct, supported by the evidence, "
            "and free of invented facts. Then fill the `answer_review` block in both "
            "JSON reports (status, reviewer, citation_accuracy, hallucination_rate)."
        ),
        "",
        "| # | Case | Kind | Offline | Online |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(offline["rows"], 1):
        anchor = f"{i}-{r['case_id'].replace('_', '-')}"
        o = rows_on[r["case_id"]]
        lines.append(
            f"| {i} | [{r['case_id']}](#{anchor}) | {r['kind']} | {_tag(r)} | {_tag(o)} |"
        )
    lines.append("")
    for i, r in enumerate(offline["rows"], 1):
        cid = r["case_id"]
        lab = by_id.get(cid, {})
        lines += [f"## {i}. {cid}", "", f"**Question:** {r['question']}", ""]
        if lab.get("history"):
            lines.append("**Prior turns:**")
            for h in lab["history"]:
                lines.append(f"- {h.get('role', '?')}: {h.get('content', h)}")
            lines.append("")
        meta = [f"labelled pages {lab.get('relevant_pages') or 'none'}"]
        if "answerable" in lab:
            meta.append(f"answerable from book: {lab['answerable']}")
        meta.append(f"web expected: {lab.get('web_expected', 'MISSING')}")
        if lab.get("expected_check_status"):
            meta.append(f"expected check: {lab['expected_check_status']}")
        lines += ["**Label:** " + " · ".join(meta), ""]
        if lab.get("expected_claims"):
            lines.append("**Expected claims:**")
            for c in lab["expected_claims"]:
                pages = ", ".join(map(str, c.get("source_pages", [])))
                lines.append(f"- {c['claim']} (p. {pages})")
            lines.append("")
        lines += _side("Offline (web off)", r, lab)
        lines += _side("Online (web on)", rows_on[cid], lab)
        lines += [
            REVIEW_BOXES,
            "",
            "Notes:",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline", type=Path)
    parser.add_argument("online", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists; refusing to overwrite")
    text = build(
        json.loads(args.offline.read_text()),
        json.loads(args.online.read_text()),
        args.offline,
        args.online,
    )
    args.output.write_text(text)
    print(args.output)
