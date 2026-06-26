"""Unit tests for SafeExpressionEvaluator — Phase 3 checkpoint.

Tests cover all 3 simple POC formulas (penale_contrattuale, interessi_mora, imu)
plus AST security rejection cases.

TFR is NOT tested here because it requires live DB access (ISTAT coefficients).
TFR correctness is verified after seeding phase via a manual integration test.

Run:
    python -m pytest tests/test_executor.py -v
    # or run as a plain script:
    python tests/test_executor.py
"""

import pytest
from src.calculator.executor import SafeExpressionEvaluator


# ---------------------------------------------------------------------------
# Penale contrattuale
# Art. 113-bis D.Lgs. 50/2016
# ---------------------------------------------------------------------------

PENALE_EXPR = (
    "min(importo_contrattuale * massimale, "
    "max(0, giorni_ritardo - giorni_franchigia) "
    "* importo_contrattuale * aliquota_giornaliera)"
)


def test_penale_below_massimale():
    """18 days delay on €120k contract at 0.5%/day → penale = €9.000."""
    ev = SafeExpressionEvaluator({
        "importo_contrattuale": 120000,
        "giorni_ritardo": 18,
        "giorni_franchigia": 3,
        "aliquota_giornaliera": 0.005,
        "massimale": 0.10
    })
    # (18 - 3) * 120000 * 0.005 = 9000 < 12000 (massimale) → 9000
    assert ev.eval(PENALE_EXPR) == 9000.0


def test_penale_massimale_applies():
    """250 days delay: computed penale exceeds 10% cap → massimale = €12.000."""
    ev = SafeExpressionEvaluator({
        "importo_contrattuale": 120000,
        "giorni_ritardo": 250,
        "giorni_franchigia": 3,
        "aliquota_giornaliera": 0.005,
        "massimale": 0.10
    })
    # (250 - 3) * 120000 * 0.005 = 148200 > 12000 (massimale) → 12000
    assert ev.eval(PENALE_EXPR) == 12000.0


# ---------------------------------------------------------------------------
# Interessi di mora
# Art. 1284 c.c.; D.Lgs. 231/2002
# ---------------------------------------------------------------------------

MORA_EXPR = "importo_dovuto * (tasso_annuo / 365) * giorni_ritardo"


def test_interessi_mora():
    """€50k capital, 5% annual rate, 90 days delay → €616.44."""
    ev = SafeExpressionEvaluator({
        "importo_dovuto": 50000,
        "giorni_ritardo": 90,
        "tasso_annuo": 0.05
    })
    # 50000 * (0.05 / 365) * 90 = 616.4383...
    result = round(ev.eval(MORA_EXPR), 2)
    assert result == 616.44, f"mora failed: {result}"


# ---------------------------------------------------------------------------
# IMU — Imposta Municipale Unica
# Art. 1 cc. 739-783 L. 160/2019
# ---------------------------------------------------------------------------

IMU_EXPR = (
    "0 if is_abitazione_principale else "
    "max(12.0, (base_imponibile * (0.5 if is_immobile_storico else 1.0)) * aliquota)"
)


def test_imu_standard():
    """Standard property, non-exempt, non-historic at 0.86% → €1.720."""
    ev = SafeExpressionEvaluator({
        "base_imponibile": 200000,
        "is_abitazione_principale": False,
        "is_immobile_storico": False,
        "aliquota": 0.0086
    })
    # max(12, 200000 * 1.0 * 0.0086) = max(12, 1720) = 1720
    assert ev.eval(IMU_EXPR) == 1720.0


def test_imu_abitazione_principale_exempt():
    """Primary residence — fully exempt regardless of other params → 0."""
    ev = SafeExpressionEvaluator({
        "base_imponibile": 200000,
        "is_abitazione_principale": True,
        "is_immobile_storico": False,
        "aliquota": 0.0086
    })
    assert ev.eval(IMU_EXPR) == 0


def test_imu_immobile_storico_50pct_reduction():
    """Historic property gets 50% reduction on tax base → €860."""
    ev = SafeExpressionEvaluator({
        "base_imponibile": 200000,
        "is_abitazione_principale": False,
        "is_immobile_storico": True,
        "aliquota": 0.0086
    })
    # max(12, 200000 * 0.5 * 0.0086) = max(12, 860) = 860
    assert ev.eval(IMU_EXPR) == 860.0


def test_imu_minimum_threshold():
    """Very low tax base: computed IMU falls below €12 minimum → €12."""
    ev = SafeExpressionEvaluator({
        "base_imponibile": 100,
        "is_abitazione_principale": False,
        "is_immobile_storico": False,
        "aliquota": 0.0086
    })
    # max(12, 100 * 1.0 * 0.0086) = max(12, 0.86) = 12
    assert ev.eval(IMU_EXPR) == 12.0


# ---------------------------------------------------------------------------
# AST security — all must raise, none must succeed
# ---------------------------------------------------------------------------

DANGEROUS_EXPRESSIONS = [
    ("__import__('os').system('ls')",  "import via __import__"),
    ("open('/etc/passwd').read()",      "attribute access on built-in"),
    ("'string literal'",                "bare string constant"),
    ("obj.method()",                    "attribute method call"),
]


@pytest.mark.parametrize("expr, description", DANGEROUS_EXPRESSIONS)
def test_ast_security_rejects(expr: str, description: str):
    """Dangerous expressions must raise ValueError or AttributeError."""
    with pytest.raises((ValueError, AttributeError)):
        SafeExpressionEvaluator({}).eval(expr)


# ---------------------------------------------------------------------------
# Allow running as a plain script (roadmap-compatible)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_penale_below_massimale()
    test_penale_massimale_applies()
    test_interessi_mora()
    test_imu_standard()
    test_imu_abitazione_principale_exempt()
    test_imu_immobile_storico_50pct_reduction()
    test_imu_minimum_threshold()

    for expr, desc in DANGEROUS_EXPRESSIONS:
        try:
            SafeExpressionEvaluator({}).eval(expr)
            raise AssertionError(f"Should have raised for: {desc!r} → {expr!r}")
        except (ValueError, AttributeError):
            pass

    print("All executor tests passed")
