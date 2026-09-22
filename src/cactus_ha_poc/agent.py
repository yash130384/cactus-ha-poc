"""NeedleAgent core: Orchestrates Needle 3 inference, entity resolution, and Home Assistant execution."""

from dataclasses import dataclass
import os
import time
from typing import Any, Optional
import needle

from .config import Config, load_config
from .ha_client import BaseHomeAssistantClient, create_ha_client
from .resolver import EntityResolver
from .tools import control_light, get_weather


@dataclass
class ExecutionResult:
    """Result of processing a natural language prompt."""
    success: bool
    prompt: str
    message: str
    tool_call: Optional[dict[str, Any]] = None
    entity_id: Optional[str] = None
    action: Optional[str] = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    ha_result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class NeedleAgent:
    """Agent orchestrating Needle 3 Tool Calling with Home Assistant."""

    SYSTEM_PROMPT = (
        "Du bist ein intelligenter Smart-Home-Assistent für Home Assistant. "
        "Du steuerst Lampen ('an', 'ein', 'anschalten', 'anmachen' -> on; 'aus', 'ausschalten', 'ausmachen' -> off; 'dimmen' -> dim) "
        "und beantwortest Wetterfragen. Verwende immer die passenden Tools."
    )

    CONDITION_TRANSLATIONS = {
        "partlycloudy": "teilweise bewölkt",
        "sunny": "sonnig",
        "clear": "klar",
        "clear-night": "klare Nacht",
        "cloudy": "bewölkt",
        "rainy": "regnerisch",
        "pouring": "starker Regen",
        "snowy": "schneereich",
        "windy": "windig",
        "fog": "neblig",
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        ha_client: Optional[BaseHomeAssistantClient] = None,
        resolver: Optional[EntityResolver] = None,
    ) -> None:
        self.config = config or load_config()
        # Enforce telemetry setting
        os.environ["NEEDLE_TELEMETRY"] = str(self.config.needle_telemetry)

        self.ha_client = ha_client or create_ha_client(self.config)
        self.resolver = resolver or EntityResolver()

        # Initialize Needle 3 model instance
        self.needle = needle.Needle(
            tools=[control_light, get_weather],
            system=self.SYSTEM_PROMPT,
        )

    def process_prompt(self, prompt: str, max_new_tokens: int = 64) -> ExecutionResult:
        """Process a natural language user prompt through Needle 3 and execute resulting actions."""
        start_time = time.perf_counter()

        # Prevent multi-turn history bleed (Fallstrick 3)
        self.needle.reset()

        # Run inference
        response = self.needle.complete(prompt, max_new_tokens=max_new_tokens)

        calls = response.get("function_calls") or []
        confidence = float(response.get("confidence") or 0.0)

        if not calls:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                success=False,
                prompt=prompt,
                message="Kein passender Home Assistant Befehl erkannt.",
                confidence=confidence,
                latency_ms=latency_ms,
                error="no_tool_call",
            )

        tool_call = calls[0]
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments") or {}

        # Confidence Gate Check
        if confidence < self.config.confidence_threshold:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                success=False,
                prompt=prompt,
                message=f"Befehl abgelehnt: Erkennungs-Konfidenz ({confidence:.2f}) liegt unter Schwellenwert ({self.config.confidence_threshold:.2f}).",
                tool_call=tool_call,
                confidence=confidence,
                latency_ms=latency_ms,
                error="low_confidence",
            )

        # Dispatch tool call
        if tool_name == "control_light":
            return self._handle_control_light(prompt, tool_call, arguments, confidence, start_time)
        elif tool_name == "get_weather":
            return self._handle_get_weather(prompt, tool_call, arguments, confidence, start_time)
        else:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                success=False,
                prompt=prompt,
                message=f"Unbekanntes Tool aufgerufen: '{tool_name}'",
                tool_call=tool_call,
                confidence=confidence,
                latency_ms=latency_ms,
                error="unknown_tool",
            )

    def _handle_control_light(
        self,
        prompt: str,
        tool_call: dict[str, Any],
        arguments: dict[str, Any],
        confidence: float,
        start_time: float,
    ) -> ExecutionResult:
        name = arguments.get("name", "")
        action = arguments.get("action", "on")
        brightness = arguments.get("brightness", 100)

        # Resolve colloquial name to technical Home Assistant entity_id
        entity_id = self.resolver.resolve_light(name)
        if not entity_id:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                success=False,
                prompt=prompt,
                message=f"Konnte kein passendes Licht für '{name}' finden.",
                tool_call=tool_call,
                confidence=confidence,
                latency_ms=latency_ms,
                error="entity_not_found",
            )

        # Execute light action via HA client
        pct = brightness if (action == "dim" or (action == "on" and brightness != 100)) else None
        ha_res = self.ha_client.set_light(entity_id=entity_id, action=action, brightness_pct=pct)

        # Format user response
        friendly = self.ha_client.get_state(entity_id).get("attributes", {}).get("friendly_name", name)
        new_state = ha_res.get("state", action)
        new_pct = ha_res.get("brightness_pct", brightness)

        if action == "off":
            message = f"Licht '{friendly}' ausgeschaltet (Status: {new_state})."
        elif action == "dim":
            message = f"Licht '{friendly}' auf {new_pct}% gedimmt (Status: {new_state})."
        else:
            message = f"Licht '{friendly}' eingeschaltet (Status: {new_state}, Helligkeit: {new_pct}%)."

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return ExecutionResult(
            success=True,
            prompt=prompt,
            message=message,
            tool_call=tool_call,
            entity_id=entity_id,
            action=action,
            confidence=confidence,
            latency_ms=latency_ms,
            ha_result=ha_res,
        )

    def _handle_get_weather(
        self,
        prompt: str,
        tool_call: dict[str, Any],
        arguments: dict[str, Any],
        confidence: float,
        start_time: float,
    ) -> ExecutionResult:
        location = arguments.get("location") or "Norderstedt"
        query_type = arguments.get("query_type", "all")

        # Resolve location to technical weather entity_id
        entity_id = self.resolver.resolve_weather(location)
        weather_info = self.ha_client.get_weather_info(entity_id=entity_id)

        friendly = weather_info.get("friendly_name", "Home")
        temp = weather_info.get("temperature", 17.0)
        cond_raw = weather_info.get("condition", "partlycloudy")
        cond_de = self.CONDITION_TRANSLATIONS.get(cond_raw, cond_raw)
        wind = weather_info.get("wind_speed", 14.0)
        precip = weather_info.get("precipitation", 0.0)

        # Context-aware German answers
        prompt_lower = prompt.lower()
        if "regenschirm" in prompt_lower:
            if precip > 0:
                message = f"Ja, nimm einen Regenschirm mit! In Norderstedt regnet es aktuell (Niederschlag: {precip} mm)."
            else:
                message = f"Nein, du brauchst aktuell keinen Regenschirm draußen (Niederschlag: {precip} mm, {cond_de})."
        elif any(k in prompt_lower for k in ("regn", "regen", "niederschlag")) or query_type == "rain":
            if precip > 0:
                message = f"Ja, es regnet in Norderstedt (Niederschlag: {precip} mm, {cond_de})."
            else:
                message = f"Nein, in Norderstedt regnet es zurzeit nicht (Niederschlag: {precip} mm, {cond_de})."
        elif "wind" in prompt_lower or query_type == "wind":
            message = f"Der Wind in Norderstedt weht mit {wind} km/h ({cond_de})."
        elif "temperatur" in prompt_lower or query_type == "temperature":
            message = f"Die aktuelle Temperatur in Norderstedt beträgt {temp}°C ({cond_de})."
        else:
            message = f"Wetter in Norderstedt: {cond_de}, {temp}°C, Wind: {wind} km/h, Niederschlag: {precip} mm."

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return ExecutionResult(
            success=True,
            prompt=prompt,
            message=message,
            tool_call=tool_call,
            entity_id=entity_id,
            action="get_weather",
            confidence=confidence,
            latency_ms=latency_ms,
            ha_result=weather_info,
        )
