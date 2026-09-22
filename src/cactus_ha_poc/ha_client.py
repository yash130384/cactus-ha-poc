"""Home Assistant REST API client with Live and Mock implementations."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
from typing import Any, Optional
import requests

from .config import Config

logger = logging.getLogger(__name__)


class BaseHomeAssistantClient(ABC):
    """Abstract base class for Home Assistant clients."""

    @abstractmethod
    def get_state(self, entity_id: str) -> dict[str, Any]:
        """Fetch the current state and attributes for an entity."""
        pass

    @abstractmethod
    def call_service(
        self,
        domain: str,
        service: str,
        service_data: dict[str, Any],
        return_response: bool = False,
    ) -> dict[str, Any]:
        """Execute a service call in Home Assistant."""
        pass

    def set_light(self, entity_id: str, action: str, brightness_pct: Optional[int] = None) -> dict[str, Any]:
        """Control a light entity (on, off, or dim)."""
        action_normalized = action.lower().strip()
        if action_normalized in ("on", "an", "ein"):
            service = "turn_on"
            data: dict[str, Any] = {"entity_id": entity_id}
            if brightness_pct is not None:
                data["brightness_pct"] = max(0, min(100, int(brightness_pct)))
            res = self.call_service("light", service, data)
            state = self.get_state(entity_id)
            return {
                "success": True,
                "entity_id": entity_id,
                "action": "on",
                "state": state.get("state", "on"),
                "brightness_pct": state.get("attributes", {}).get("brightness_pct", brightness_pct or 100),
                "raw": res,
            }
        elif action_normalized in ("off", "aus"):
            service = "turn_off"
            data = {"entity_id": entity_id}
            res = self.call_service("light", service, data)
            state = self.get_state(entity_id)
            return {
                "success": True,
                "entity_id": entity_id,
                "action": "off",
                "state": state.get("state", "off"),
                "brightness_pct": 0,
                "raw": res,
            }
        elif action_normalized in ("dim", "dimmen"):
            pct = 100 if brightness_pct is None else max(0, min(100, int(brightness_pct)))
            service = "turn_on"
            data = {"entity_id": entity_id, "brightness_pct": pct}
            res = self.call_service("light", service, data)
            state = self.get_state(entity_id)
            return {
                "success": True,
                "entity_id": entity_id,
                "action": "dim",
                "state": state.get("state", "on"),
                "brightness_pct": pct,
                "raw": res,
            }
        else:
            raise ValueError(f"Unsupported light action '{action}'. Expected 'on', 'off', or 'dim'.")

    def get_weather_info(self, entity_id: str = "weather.forecast_home") -> dict[str, Any]:
        """Fetch and structure weather information."""
        state = self.get_state(entity_id)
        attributes = state.get("attributes", {})
        temp = attributes.get("temperature", 17.0)
        temp_unit = attributes.get("temperature_unit", "°C")
        wind_speed = attributes.get("wind_speed", 14.0)
        wind_unit = attributes.get("wind_speed_unit", "km/h")
        precip = attributes.get("precipitation", 0.0)
        precip_unit = attributes.get("precipitation_unit", "mm")
        condition = state.get("state", attributes.get("condition", "partlycloudy"))
        friendly_name = attributes.get("friendly_name", "Home")
        forecast = attributes.get("forecast", [])

        # In modern HA, forecasts might be retrieved via weather.get_forecasts service
        if not forecast:
            try:
                forecast_res = self.call_service(
                    "weather",
                    "get_forecasts",
                    {"entity_id": entity_id, "type": "daily"},
                    return_response=True,
                )
                if isinstance(forecast_res, dict):
                    service_resp = forecast_res.get("service_response", forecast_res)
                    if isinstance(service_resp, dict) and entity_id in service_resp:
                        forecast = service_resp[entity_id].get("forecast", [])
            except Exception as e:
                logger.debug(f"Could not fetch weather forecast service for {entity_id}: {e}")

        return {
            "entity_id": entity_id,
            "friendly_name": friendly_name,
            "state": condition,
            "condition": condition,
            "temperature": temp,
            "temperature_unit": temp_unit,
            "wind_speed": wind_speed,
            "wind_speed_unit": wind_unit,
            "precipitation": precip,
            "precipitation_unit": precip_unit,
            "forecast": forecast,
        }


class MockHomeAssistantClient(BaseHomeAssistantClient):
    """Simulated Home Assistant REST Client for testing and offline operation."""

    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.reset_defaults()

    def reset_defaults(self) -> None:
        """Reset mock entities to initial state."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self.entities = {
            "light.esstisch": {
                "entity_id": "light.esstisch",
                "state": "off",
                "attributes": {
                    "friendly_name": "Esstisch",
                    "brightness": 0,
                    "brightness_pct": 0,
                    "supported_color_modes": ["brightness", "color_temp"],
                },
                "last_changed": now_iso,
                "last_updated": now_iso,
            },
            "light.kuche": {
                "entity_id": "light.kuche",
                "state": "off",
                "attributes": {
                    "friendly_name": "Küche",
                    "brightness": 0,
                    "brightness_pct": 0,
                    "supported_color_modes": ["onoff"],
                },
                "last_changed": now_iso,
                "last_updated": now_iso,
            },
            "light.flur_oben": {
                "entity_id": "light.flur_oben",
                "state": "off",
                "attributes": {
                    "friendly_name": "Flur oben",
                    "brightness": 0,
                    "brightness_pct": 0,
                },
                "last_changed": now_iso,
                "last_updated": now_iso,
            },
            "weather.forecast_home": {
                "entity_id": "weather.forecast_home",
                "state": "partlycloudy",
                "attributes": {
                    "friendly_name": "Home",
                    "temperature": 17.0,
                    "temperature_unit": "°C",
                    "condition": "partlycloudy",
                    "wind_speed": 14.0,
                    "wind_speed_unit": "km/h",
                    "precipitation": 0.0,
                    "precipitation_unit": "mm",
                    "humidity": 65,
                    "forecast": [
                        {
                            "datetime": "2026-09-23T12:00:00+00:00",
                            "condition": "partlycloudy",
                            "temperature": 19.0,
                            "templow": 11.0,
                            "precipitation": 0.0,
                            "wind_speed": 15.0,
                        }
                    ],
                },
                "last_changed": now_iso,
                "last_updated": now_iso,
            },
        }

    def get_state(self, entity_id: str) -> dict[str, Any]:
        if entity_id in self.entities:
            return self.entities[entity_id]
        raise ValueError(f"Entity '{entity_id}' not found in mock registry.")

    def call_service(
        self,
        domain: str,
        service: str,
        service_data: dict[str, Any],
        return_response: bool = False,
    ) -> dict[str, Any]:
        entity_id = service_data.get("entity_id")
        now_iso = datetime.now(timezone.utc).isoformat()

        if domain == "light":
            if entity_id not in self.entities:
                # Dynamically register if missing
                self.entities[entity_id] = {
                    "entity_id": entity_id,
                    "state": "off",
                    "attributes": {"friendly_name": entity_id.split(".", 1)[-1], "brightness": 0, "brightness_pct": 0},
                    "last_changed": now_iso,
                    "last_updated": now_iso,
                }
            entity = self.entities[entity_id]

            if service == "turn_on":
                pct = service_data.get("brightness_pct")
                if pct is None and service_data.get("brightness"):
                    pct = int(round(service_data["brightness"] / 255.0 * 100))
                if pct is None:
                    pct = 100
                entity["state"] = "on"
                entity["attributes"]["brightness_pct"] = pct
                entity["attributes"]["brightness"] = int(round(pct / 100.0 * 255))
                entity["last_changed"] = now_iso
                entity["last_updated"] = now_iso
                return {"success": True, "entity": entity}

            elif service == "turn_off":
                entity["state"] = "off"
                entity["attributes"]["brightness_pct"] = 0
                entity["attributes"]["brightness"] = 0
                entity["last_changed"] = now_iso
                entity["last_updated"] = now_iso
                return {"success": True, "entity": entity}

            else:
                raise ValueError(f"Unsupported light service: {service}")

        elif domain == "weather" and service == "get_forecasts":
            target = entity_id or "weather.forecast_home"
            forecast = self.entities.get(target, {}).get("attributes", {}).get("forecast", [])
            if return_response:
                return {"changed_states": [], "service_response": {target: {"forecast": forecast}}}
            return {target: {"forecast": forecast}}

        return {"success": True, "domain": domain, "service": service, "data": service_data}


class LiveHomeAssistantClient(BaseHomeAssistantClient):
    """Live Home Assistant REST API Client."""

    def __init__(self, base_url: str, token: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def check_api(self) -> bool:
        """Check if Home Assistant API is responsive and authenticated."""
        try:
            resp = self.session.get(f"{self.base_url}/api/", timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def get_state(self, entity_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/states/{entity_id}"
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code == 404:
            raise ValueError(f"Entity '{entity_id}' not found in Home Assistant.")
        resp.raise_for_status()
        return resp.json()

    def call_service(
        self,
        domain: str,
        service: str,
        service_data: dict[str, Any],
        return_response: bool = False,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/api/services/{domain}/{service}"
        request_options = {"params": {"return_response": "true"}} if return_response else {}
        resp = self.session.post(url, json=service_data, timeout=self.timeout, **request_options)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"status": resp.status_code, "text": resp.text}


def create_ha_client(config: Optional[Config] = None) -> BaseHomeAssistantClient:
    """Factory creating LiveHomeAssistantClient or MockHomeAssistantClient based on Config."""
    if config is None:
        from .config import load_config
        config = load_config()

    if config.mock_mode or not config.hass_token:
        logger.info("Initializing Mock Home Assistant Client.")
        return MockHomeAssistantClient()

    logger.info(f"Initializing Live Home Assistant Client at {config.hass_url}.")
    return LiveHomeAssistantClient(base_url=config.hass_url, token=config.hass_token)
