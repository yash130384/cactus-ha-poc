"""Unit tests for Home Assistant client (Mock and Live)."""

from unittest.mock import MagicMock, patch
import pytest
from cactus_ha_poc.config import Config
from cactus_ha_poc.ha_client import (
    LiveHomeAssistantClient,
    MockHomeAssistantClient,
    create_ha_client,
)


def test_mock_initial_states():
    client = MockHomeAssistantClient()
    esstisch = client.get_state("light.esstisch")
    assert esstisch["state"] == "off"
    assert esstisch["attributes"]["friendly_name"] == "Esstisch"
    assert esstisch["attributes"]["brightness_pct"] == 0

    weather = client.get_state("weather.forecast_home")
    assert weather["state"] == "partlycloudy"
    assert weather["attributes"]["temperature"] == 17.0
    assert weather["attributes"]["precipitation"] == 0.0
    assert weather["attributes"]["wind_speed"] == 14.0


def test_mock_set_light_turn_on():
    client = MockHomeAssistantClient()
    res = client.set_light("light.esstisch", action="on")
    assert res["success"] is True
    assert res["state"] == "on"
    assert res["brightness_pct"] == 100

    state = client.get_state("light.esstisch")
    assert state["state"] == "on"
    assert state["attributes"]["brightness_pct"] == 100
    assert state["attributes"]["brightness"] == 255


def test_mock_set_light_turn_off():
    client = MockHomeAssistantClient()
    # First turn on
    client.set_light("light.esstisch", action="on")
    # Then turn off
    res = client.set_light("light.esstisch", action="off")
    assert res["success"] is True
    assert res["state"] == "off"
    assert res["brightness_pct"] == 0

    state = client.get_state("light.esstisch")
    assert state["state"] == "off"
    assert state["attributes"]["brightness_pct"] == 0


def test_mock_set_light_dim():
    client = MockHomeAssistantClient()
    res = client.set_light("light.esstisch", action="dim", brightness_pct=40)
    assert res["success"] is True
    assert res["state"] == "on"
    assert res["brightness_pct"] == 40

    state = client.get_state("light.esstisch")
    assert state["state"] == "on"
    assert state["attributes"]["brightness_pct"] == 40
    assert state["attributes"]["brightness"] == int(round(40 / 100.0 * 255))


def test_mock_german_action_aliases():
    client = MockHomeAssistantClient()
    res_an = client.set_light("light.kuche", action="an")
    assert res_an["state"] == "on"

    res_aus = client.set_light("light.kuche", action="aus")
    assert res_aus["state"] == "off"


def test_mock_invalid_action():
    client = MockHomeAssistantClient()
    with pytest.raises(ValueError, match="Unsupported light action"):
        client.set_light("light.esstisch", action="explode")


def test_mock_weather_info():
    client = MockHomeAssistantClient()
    weather = client.get_weather_info("weather.forecast_home")
    assert weather["entity_id"] == "weather.forecast_home"
    assert weather["temperature"] == 17.0
    assert weather["precipitation"] == 0.0
    assert weather["wind_speed"] == 14.0
    assert weather["condition"] == "partlycloudy"
    assert len(weather["forecast"]) >= 1


def test_create_ha_client_factory_mock():
    cfg_mock = Config(mock_mode=True, hass_token=None)
    client = create_ha_client(cfg_mock)
    assert isinstance(client, MockHomeAssistantClient)

    cfg_no_token = Config(mock_mode=False, hass_token=None)
    client2 = create_ha_client(cfg_no_token)
    assert isinstance(client2, MockHomeAssistantClient)


def test_create_ha_client_factory_live():
    cfg_live = Config(mock_mode=False, hass_token="test_token", hass_url="http://ha.local:8123")
    client = create_ha_client(cfg_live)
    assert isinstance(client, LiveHomeAssistantClient)
    assert client.base_url == "http://ha.local:8123"
    assert client.token == "test_token"


def test_live_client_calls():
    client = LiveHomeAssistantClient(base_url="http://127.0.0.1:8123", token="mock_secret_token")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "entity_id": "light.esstisch",
        "state": "on",
        "attributes": {"brightness_pct": 50, "brightness": 128},
    }

    with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
        state = client.get_state("light.esstisch")
        mock_get.assert_called_once_with("http://127.0.0.1:8123/api/states/light.esstisch", timeout=5.0)
        assert state["state"] == "on"

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        res = client.call_service("light", "turn_on", {"entity_id": "light.esstisch", "brightness_pct": 50})
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8123/api/services/light/turn_on",
            json={"entity_id": "light.esstisch", "brightness_pct": 50},
            timeout=5.0,
        )
        assert res["entity_id"] == "light.esstisch"

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        client.call_service("weather", "get_forecasts", {"entity_id": "weather.forecast_home", "type": "daily"}, return_response=True)
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8123/api/services/weather/get_forecasts",
            json={"entity_id": "weather.forecast_home", "type": "daily"},
            params={"return_response": "true"},
            timeout=5.0,
        )
