"""Conservative trusted symbolic checks; imported inside the sandbox only.

Model or user text never reaches ``eval``/``sympify``/``parse_expr``. A bounded AST
is translated into SymPy constructors over exact numbers and one real symbol.
Variable denominators are recorded while parsing (before SymPy can cancel them),
so domain-sensitive identities stay unverified instead of being proved by
cancellation. This module is standalone: limits arrive from the host.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass

KINDS = ("substitution", "derivative", "equivalence")
FUNCTIONS = ("sin", "cos", "exp")
_VARIABLE = re.compile(r"[a-z]")


@dataclass(frozen=True)
class VerificationCheck:
    kind: str
    expression: str
    candidate: str
    variable: str


@dataclass(frozen=True)
class CheckResult:
    status: str  # verified | disagreed | unverified | error
    reason: str


class Unsupported(ValueError):
    """Outside the conservative grammar; the check is declined, not failed."""


class _Builder:
    def __init__(self, source: str, symbol, limits: dict):
        import sympy

        self.sympy = sympy
        self.source = source
        self.symbol = symbol
        self.limits = limits
        self.denominators = []

    def build(self, node):
        sympy = self.sympy
        if isinstance(node, ast.Expression):
            return self.build(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise Unsupported("Only numeric literals are supported")
            text = ast.get_source_segment(self.source, node) or ""
            digits = sum(ch.isdigit() for ch in text)
            if (
                digits > self.limits["digits"]
                or "e" in text.lower()
                or "j" in text.lower()
            ):
                raise Unsupported("Numeric literal outside supported range")
            return sympy.Rational(text)  # exact, never a rounded float
        if isinstance(node, ast.Name):
            if node.id != self.symbol.name:
                raise Unsupported(f"Unknown name: {node.id}")
            return self.symbol
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.build(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left = self.build(node.left)
            if isinstance(node.op, ast.Pow):
                exponent = self.build(node.right)
                if not (
                    exponent.is_Integer
                    and abs(int(exponent)) <= self.limits["exponent"]
                ):
                    raise Unsupported("Only bounded integer exponents are supported")
                if exponent < 0 and left.has(self.symbol):
                    self.denominators.append(left)
                return sympy.Pow(left, exponent)
            right = self.build(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right.has(self.symbol):
                    self.denominators.append(right)
                elif right == 0:
                    raise Unsupported("Division by zero")
                return left / right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            return getattr(sympy, node.func.id)(self.build(node.args[0]))
        raise Unsupported(f"Unsupported syntax: {type(node).__name__}")


def parse(text: str, symbol, limits: dict):
    if not isinstance(text, str) or not text.strip():
        raise Unsupported("Empty expression")
    if len(text) > limits["chars"]:
        raise Unsupported("Expression too long")
    text = text.strip().replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise Unsupported("Expression is not valid arithmetic") from exc
    if sum(1 for _ in ast.walk(tree)) > limits["nodes"]:
        raise Unsupported("Expression too complex")
    builder = _Builder(text, symbol, limits)
    return builder.build(tree), builder.denominators


def _zero(value, sympy) -> bool | None:
    """True if provably zero, False if provably nonzero, None if undecided."""
    simplified = sympy.simplify(value)
    if simplified.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
        return None
    if simplified == 0:
        return True
    if simplified.is_zero is False or (simplified.is_number and simplified != 0):
        return False
    return None


# Exact rational sample points; any defined nonzero value disproves an identity.
_SAMPLES = ("1/3", "7/5", "2", "-3/7", "11/4")


def _identity_counterexample(difference, denominators, symbol, sympy):
    for text in _SAMPLES:
        point = sympy.Rational(text)
        if any(d.subs(symbol, point) == 0 for d in denominators):
            continue
        value = sympy.simplify(difference.subs(symbol, point))
        if value.is_number and value.is_finite and value != 0:
            return False
    return None


def evaluate_check(check: VerificationCheck, limits: dict) -> CheckResult:
    import sympy

    if check.kind not in KINDS:
        return CheckResult("error", f"Unknown check kind: {check.kind}")
    if not isinstance(check.variable, str) or not _VARIABLE.fullmatch(check.variable):
        return CheckResult("error", "Variable must be one lowercase letter")
    symbol = sympy.Symbol(check.variable, real=True)
    try:
        expression, expr_denominators = parse(check.expression, symbol, limits)
        candidate, cand_denominators = parse(check.candidate, symbol, limits)
    except Unsupported as exc:
        return CheckResult("unverified", f"Declined: {exc}")
    denominators = expr_denominators + cand_denominators
    if check.kind == "substitution":
        if candidate.has(symbol):
            return CheckResult("unverified", "Declined: candidate must be a number")
        for denominator in denominators:
            if _zero(denominator.subs(symbol, candidate), sympy) is not False:
                return CheckResult(
                    "unverified", "Candidate is outside the expression's domain"
                )
        verdict = _zero(expression.subs(symbol, candidate), sympy)
        if verdict is None:
            return CheckResult("unverified", "Residual could not be decided exactly")
        return CheckResult(
            "verified" if verdict else "disagreed",
            f"Residual at {check.variable}={candidate} is {'zero' if verdict else 'nonzero'}",
        )
    if check.kind == "equivalence" and denominators:
        return CheckResult(
            "unverified",
            "Declined: variable denominators make equivalence domain-sensitive",
        )
    target = (
        sympy.diff(expression, symbol) if check.kind == "derivative" else expression
    )
    difference = target - candidate
    verdict = _zero(difference, sympy)
    if verdict is None:
        verdict = _identity_counterexample(difference, denominators, symbol, sympy)
    if verdict is None:
        return CheckResult("unverified", "Difference could not be decided exactly")
    label = "derivative" if check.kind == "derivative" else "expressions"
    return CheckResult(
        "verified" if verdict else "disagreed",
        f"The {label} {'match' if verdict else 'differ'} identically"
        + (" where defined" if denominators else ""),
    )


def run_check(payload: dict, limits: dict) -> str:
    """Sandbox entry point: JSON in, JSON out; never raises to the caller."""
    try:
        check = VerificationCheck(**payload)
        result = evaluate_check(check, limits)
    except (TypeError, ValueError, RecursionError, MemoryError, OverflowError) as exc:
        result = CheckResult("error", f"{type(exc).__name__}")
    return json.dumps(asdict(result))
