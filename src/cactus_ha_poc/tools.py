"""Needle 3 tool definitions for Home Assistant smart home control."""

from typing import Literal
import needle


@needle.tool
def control_light(
    name: str,
    action: Literal["on", "off", "dim"],
    brightness: int = 100,
) -> str:
    """Controls a light or lamp in Home Assistant.

    Args:
        name: Name of the light or room (e.g. 'Esstisch', 'Esstischlicht', 'Küche', 'Flur').
        action: Operation to perform. 'on' (an/ein/anschalten/anmachen), 'off' (aus/ausschalten/ausmachen), or 'dim' (dimmen/Helligkeit anpassen).
        brightness: Brightness level from 0 to 100 percent when turning on or dimming (default: 100).
    """
    return f"control_light(name={name!r}, action={action!r}, brightness={brightness})"


@needle.tool
def get_weather(
    location: str = "Norderstedt",
    query_type: Literal["all", "temperature", "rain", "wind"] = "all",
) -> str:
    """Retrieves current weather conditions or forecasts from Home Assistant.

    Args:
        location: City or location name. Defaults to 'Norderstedt'.
        query_type: Type of weather information requested: 'all' (general conditions), 'temperature' (Temperatur), 'rain' (Regen/Niederschlag/Regenschirm), or 'wind' (Windstärke/Wind).
    """
    return f"get_weather(location={location!r}, query_type={query_type!r})"
