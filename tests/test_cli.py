"""Unit and integration tests for the CLI interface."""

from unittest.mock import patch
import pytest
from cactus_ha_poc.cli import main, run_benchmark
from cactus_ha_poc.agent import NeedleAgent
from cactus_ha_poc.ha_client import MockHomeAssistantClient
from cactus_ha_poc.config import Config


def test_cli_single_prompt_success(capsys):
    exit_code = main(["--mock", "Schalte das Licht am Esstisch an"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "ERFOLG" in captured
    assert "control_light" in captured
    assert "light.esstisch" in captured


def test_cli_single_prompt_failure(capsys):
    # Prompt with confidence below threshold
    exit_code = main(["--mock", "--threshold", "0.9999", "Wie ist das Wetter in Norderstedt?"])
    captured = capsys.readouterr().out
    assert exit_code == 1
    assert "FEHLER" in captured


def test_cli_eval_benchmark(capsys):
    cfg = Config(mock_mode=True, confidence_threshold=0.40)
    agent = NeedleAgent(config=cfg, ha_client=MockHomeAssistantClient())
    exit_code = run_benchmark(agent)
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Benchmark-Zusammenfassung" in captured
    assert "7/7" in captured
    assert "PASS" in captured


def test_cli_main_eval_flag(capsys):
    exit_code = main(["--mock", "--eval"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Benchmark-Zusammenfassung" in captured


def test_cli_interactive_quit(capsys):
    with patch("builtins.input", side_effect=["q"]):
        exit_code = main(["--mock"])
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "Cactus HA POC" in captured
        assert "Beende Cactus HA POC" in captured


def test_cli_dry_run(capsys):
    exit_code = main(["--mock", "--dry-run", "Schalte das Licht am Esstisch an"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "ERFOLG" in captured
    assert "[TEST-MODUS]" in captured
    assert "Kein physischer Schaltbefehl gesendet" in captured

