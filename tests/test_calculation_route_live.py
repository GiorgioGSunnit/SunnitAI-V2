import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from src.rag.calculation import calculation_gate, calculation_node


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIVE_URL = "http://127.0.0.1:8971"
_QUERY = (
    "calcola gli interessi legali su 8500 euro "
    "dal 01/01/2024 al 31/12/2025"
)


@pytest.fixture(scope="module", autouse=True)
def live_calculation_platform(tmp_path_factory):
    previous_url = os.environ.get("CALC_PLATFORM_URL")
    process_env = os.environ.copy()
    process_env["CALC_DB_PATH"] = str(
        tmp_path_factory.mktemp("calculation-platform") / "calculations.db"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "calculation_platform.app.main:app",
            "--port",
            "8971",
            "--log-level",
            "warning",
        ],
        cwd=_REPO_ROOT,
        env=process_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["CALC_PLATFORM_URL"] = _LIVE_URL

    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{_LIVE_URL}/health", timeout=0.5)
                if response.ok and response.json().get("status") == "ok":
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(0.1)
        else:
            pytest.skip("live calculation platform did not become healthy within 20s")

        yield
    finally:
        if previous_url is None:
            os.environ.pop("CALC_PLATFORM_URL", None)
        else:
            os.environ["CALC_PLATFORM_URL"] = previous_url
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_live_success_end_to_end():
    gate = calculation_gate({"query": _QUERY})

    assert gate["calc_route"] == "calculate"
    update = calculation_node(
        {
            **gate,
            "query": _QUERY,
            "raw_query": _QUERY,
            "session_language": "it",
        }
    )

    assert re.search(r"\d+[.,]\d+", update["answer"])
    assert "Fonti:" in update["answer"]
    assert update["pending_calculation"] is None


def test_live_missing_input_then_continuation():
    query = "calcola gli interessi legali"
    gate = calculation_gate({"query": query})

    assert gate["calc_route"] == "calculate"
    first = calculation_node(
        {
            **gate,
            "query": query,
            "raw_query": query,
            "session_language": "it",
        }
    )
    assert first.get("answer")
    assert first["pending_calculation"]

    raw_follow_up = "8500 euro dal 01/01/2024 al 31/12/2025"
    second = calculation_node(
        {
            "query": "rewritten follow-up without reliable values",
            "raw_query": raw_follow_up,
            "session_language": "it",
            "pending_calculation": first["pending_calculation"],
        }
    )

    assert re.search(r"\d+[.,]\d+", second["answer"])
    assert "Fonti:" in second["answer"]
    assert second["pending_calculation"] is None


def test_live_outage_gate_falls_back(monkeypatch):
    monkeypatch.setenv("CALC_PLATFORM_URL", "http://127.0.0.1:65534")

    assert calculation_gate({"query": _QUERY})["calc_route"] == "normal"
