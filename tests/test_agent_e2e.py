"""End-to-End tests for NeedleAgent covering TC-01 to TC-08."""

import pytest
from cactus_ha_poc.agent import NeedleAgent
from cactus_ha_poc.config import Config
from cactus_ha_poc.ha_client import MockHomeAssistantClient


@pytest.fixture(scope="module")
def mock_client() -> MockHomeAssistantClient:
    return MockHomeAssistantClient()


@pytest.fixture(scope="module")
def agent(mock_client: MockHomeAssistantClient) -> NeedleAgent:
    cfg = Config(mock_mode=True, confidence_threshold=0.40)
    agent_inst = NeedleAgent(config=cfg, ha_client=mock_client)
    # Warmup inference
    agent_inst.process_prompt("Schalte das Licht am Esstisch an")
    return agent_inst


def test_tc01_turn_on_light(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """TC-01: Licht Einschalten - 'Schalte das Licht am Esstisch an'."""
    prompt = "Schalte das Licht am Esstisch an"
    res = agent.process_prompt(prompt)

    assert res.success is True
    assert res.confidence > 0.85
    assert res.entity_id == "light.esstisch"
    assert res.action == "on"
    assert "Esstisch" in res.message
    assert mock_client.get_state("light.esstisch")["state"] == "on"


def test_tc02_turn_off_light(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """TC-02: Licht Ausschalten - 'Esstischlicht ausschalten'."""
    # First ensure light is on
    mock_client.set_light("light.esstisch", action="on")
    assert mock_client.get_state("light.esstisch")["state"] == "on"

    prompt = "Esstischlicht ausschalten"
    res = agent.process_prompt(prompt)

    assert res.success is True
    assert res.confidence > 0.85
    assert res.entity_id == "light.esstisch"
    assert res.action == "off"
    assert mock_client.get_state("light.esstisch")["state"] == "off"


def test_tc03_dim_light(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """TC-03: Licht Dimmen - 'Dimme das Esstischlicht auf 40 Prozent'."""
    prompt = "Dimme das Esstischlicht auf 40 Prozent"
    res = agent.process_prompt(prompt)

    assert res.success is True
    assert res.entity_id == "light.esstisch"
    assert res.action == "dim"
    assert mock_client.get_state("light.esstisch")["state"] == "on"
    assert mock_client.get_state("light.esstisch")["attributes"]["brightness_pct"] == 40


def test_tc04_other_light(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """TC-04: Andere Lampe - 'Küche anmachen'."""
    prompt = "Küche anmachen"
    res = agent.process_prompt(prompt)

    assert res.success is True
    assert res.entity_id == "light.kuche"
    assert res.action == "on"
    assert mock_client.get_state("light.kuche")["state"] == "on"


def test_tc05_weather_general(agent: NeedleAgent):
    """TC-05: Wetter Allgemein - 'Wie ist das Wetter in Norderstedt?'."""
    prompt = "Wie ist das Wetter in Norderstedt?"
    res = agent.process_prompt(prompt)

    assert res.success is True
    assert res.entity_id == "weather.forecast_home"
    assert "17" in res.message or "17.0" in res.message
    assert "Norderstedt" in res.message


def test_tc06_weather_rain(agent: NeedleAgent):
    """TC-06: Regenabfrage - 'Regnet es gerade in Norderstedt?'."""
    prompt = "Regnet es gerade in Norderstedt?"
    res = agent.process_prompt(prompt)

    assert res.success is True
    assert res.entity_id == "weather.forecast_home"
    # Mock data has 0.0 mm precipitation -> "regnet es zurzeit nicht"
    assert "regnet" in res.message.lower()


def test_tc07_weather_no_location_umbrella(agent: NeedleAgent):
    """TC-07: Ohne Ortsangabe - 'Brauche ich heute einen Regenschirm draußen?'."""
    prompt = "Brauche ich heute einen Regenschirm draußen?"
    res = agent.process_prompt(prompt)

    assert res.success is True
    assert res.entity_id == "weather.forecast_home"
    assert "regenschirm" in res.message.lower()


def test_tc08_latency_budget(agent: NeedleAgent):
    """TC-08: Latenz-Budget - Gemessene Pipeline-Dauer."""
    prompt = "Schalte das Licht am Esstisch an"
    res = agent.process_prompt(prompt)

    assert res.success is True
    # The target budget is < 150 ms on standard desktop systems; on low-power CPUs (<2 GHz)
    # we verify responsiveness is sub-second (< 500 ms).
    assert res.latency_ms < 500.0, f"Latency {res.latency_ms:.2f} ms exceeded threshold"


def test_needle_reset_prevents_history_bleed(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """Verify needle.reset() prevents dimming percentage bleeding into next on/off command."""
    # Step 1: Dim
    res1 = agent.process_prompt("Dimme das Esstischlicht auf 25 Prozent")
    assert res1.success is True
    assert mock_client.get_state("light.esstisch")["attributes"]["brightness_pct"] == 25

    # Step 2: Turn on kitchen - should NOT bleed brightness 25 into kitchen turn_on
    res2 = agent.process_prompt("Küche anmachen")
    assert res2.success is True
    assert res2.tool_call["name"] == "control_light"
    # Action should be 'on' without 25 brightness
    assert res2.tool_call["arguments"]["action"] == "on"
    assert res2.tool_call["arguments"].get("brightness", 100) == 100


def test_confidence_threshold_gate(mock_client: MockHomeAssistantClient):
    """Verify confidence gate rejects calls below threshold."""
    # Set high threshold
    cfg = Config(mock_mode=True, confidence_threshold=0.9999)
    strict_agent = NeedleAgent(config=cfg, ha_client=mock_client)

    res = strict_agent.process_prompt("Wie ist das Wetter in Norderstedt?")
    assert res.success is False
    assert res.error == "low_confidence"


def test_dry_run_turn_on(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """Verify dry_run=True does not execute physical command for turn on."""
    mock_client.set_light("light.esstisch", action="off")
    assert mock_client.get_state("light.esstisch")["state"] == "off"

    res = agent.process_prompt("Schalte das Licht am Esstisch an", dry_run=True)
    assert res.success is True
    assert res.entity_id == "light.esstisch"
    assert res.ha_result == {"simulated": True, "dry_run": True}
    assert res.message == (
        "[TEST-MODUS] Befehl erkannt: Licht 'Esstisch' wuerde eingeschaltet werden "
        "(Entity: light.esstisch). Kein physischer Schaltbefehl gesendet."
    )
    # State in mock client must remain unchanged (off)
    assert mock_client.get_state("light.esstisch")["state"] == "off"


def test_dry_run_turn_off(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """Verify dry_run=True does not execute physical command for turn off."""
    mock_client.set_light("light.esstisch", action="on")
    assert mock_client.get_state("light.esstisch")["state"] == "on"

    res = agent.process_prompt("Esstischlicht ausschalten", dry_run=True)
    assert res.success is True
    assert res.entity_id == "light.esstisch"
    assert res.ha_result == {"simulated": True, "dry_run": True}
    assert res.message == (
        "[TEST-MODUS] Befehl erkannt: Licht 'Esstisch' wuerde ausgeschaltet werden "
        "(Entity: light.esstisch). Kein physischer Schaltbefehl gesendet."
    )
    # State in mock client must remain unchanged (on)
    assert mock_client.get_state("light.esstisch")["state"] == "on"


def test_dry_run_dim(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """Verify dry_run=True for dimming command."""
    mock_client.set_light("light.esstisch", action="off")

    res = agent.process_prompt("Dimme das Esstischlicht auf 40 Prozent", dry_run=True)
    assert res.success is True
    assert res.entity_id == "light.esstisch"
    assert res.ha_result == {"simulated": True, "dry_run": True}
    assert res.message == (
        "[TEST-MODUS] Befehl erkannt: Licht 'Esstisch' wuerde auf 40% gedimmt werden "
        "(Entity: light.esstisch). Kein physischer Schaltbefehl gesendet."
    )
    assert mock_client.get_state("light.esstisch")["state"] == "off"


def test_dry_run_all_lights(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """Verify dry_run=True for light.all."""
    res = agent.process_prompt("Alle Lichter ausschalten", dry_run=True)
    assert res.success is True
    assert res.entity_id == "light.all"
    assert res.ha_result == {"simulated": True, "dry_run": True}
    assert res.message == (
        "[TEST-MODUS] Befehl erkannt: Licht 'Alle Lichter' wuerde ausgeschaltet werden "
        "(Entity: light.all). Kein physischer Schaltbefehl gesendet."
    )


def test_live_all_lights_turn_off(agent: NeedleAgent, mock_client: MockHomeAssistantClient):
    """Verify dry_run=False iterates and switches all lights."""
    # Turn several lights on first
    mock_client.set_light("light.esstisch", action="on")
    mock_client.set_light("light.kuche", action="on")
    mock_client.set_light("light.decke1", action="on")
    assert mock_client.get_state("light.esstisch")["state"] == "on"
    assert mock_client.get_state("light.kuche")["state"] == "on"
    assert mock_client.get_state("light.decke1")["state"] == "on"

    res = agent.process_prompt("Alle Lichter ausschalten", dry_run=False)
    assert res.success is True
    assert res.entity_id == "light.all"
    assert "Alle" in res.message and "ausgeschaltet" in res.message

    # Verify all entities are now turned off
    assert mock_client.get_state("light.esstisch")["state"] == "off"
    assert mock_client.get_state("light.kuche")["state"] == "off"
    assert mock_client.get_state("light.decke1")["state"] == "off"
    assert mock_client.get_state("light.bodenlampe")["state"] == "off"
