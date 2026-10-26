import pytest

from src import config
from src.tools.math_checks import VerificationCheck, evaluate_check, run_check

LIMITS = {
    "chars": config.MATH_MAX_EXPRESSION_CHARS,
    "nodes": config.MATH_MAX_AST_NODES,
    "exponent": config.MATH_MAX_INTEGER_EXPONENT,
    "digits": config.MATH_MAX_NUMERIC_DIGITS,
}


def check(kind, expression, candidate, variable="x"):
    return evaluate_check(
        VerificationCheck(kind, expression, candidate, variable), LIMITS
    )


@pytest.mark.parametrize(
    ("kind", "expression", "candidate", "status"),
    [
        ("substitution", "x*x-4", "2", "verified"),
        ("substitution", "x*x-4", "3", "disagreed"),
        ("substitution", "x**2 - 5*x + 6", "3", "verified"),
        ("substitution", "x^2 - 2", "1.4142", "disagreed"),
        ("derivative", "x**3", "3*x**2", "verified"),
        ("derivative", "x**3", "2*x", "disagreed"),
        ("derivative", "sin(x)*x", "cos(x)*x + sin(x)", "verified"),
        ("derivative", "1/x", "-1/x**2", "verified"),
        ("equivalence", "(x+1)**2", "x**2 + 2*x + 1", "verified"),
        ("equivalence", "(x+1)**2", "x**2 + 1", "disagreed"),
        ("equivalence", "x/x", "1", "unverified"),
        ("equivalence", "(x**2-1)/(x-1)", "x+1", "unverified"),
        ("substitution", "1/x", "0", "unverified"),
        ("substitution", "0.1 + 0.2 - x", "0.3", "verified"),
    ],
)
def test_trusted_check_regressions(kind, expression, candidate, status):
    assert check(kind, expression, candidate).status == status


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "x.__class__",
        "open('/etc/passwd')",
        "sqrt(x)",
        "y + 1",
        "x ** 65",
        "x ** 0.5",
        "lambda: 1",
        "[x]",
        "1e400",
        "1" * 40,
        "x+" * 300 + "x",
        "-" * 200 + "x",
    ],
)
def test_unsupported_input_is_declined_before_evaluation(expression):
    assert check("equivalence", expression, "x").status == "unverified"


def test_compute_payload_cannot_change_the_result():
    import json

    out = json.loads(
        run_check(
            {
                "kind": "substitution",
                "expression": "x-1",
                "candidate": "2",
                "variable": "x",
            },
            LIMITS,
        )
    )
    assert out == {"status": "disagreed", "reason": "Residual at x=2 is nonzero"}
    forged = json.loads(
        run_check(
            {
                "kind": "substitution",
                "expression": "x-1",
                "candidate": "2",
                "variable": "x",
                "verified": True,
            },
            LIMITS,
        )
    )
    assert forged["status"] == "error"


def test_invalid_kind_and_variable_are_errors():
    assert check("integral", "x", "x").status == "error"
    assert check("derivative", "x", "1", variable="xy").status == "error"
    assert check("substitution", "x-1", "x").status == "unverified"
