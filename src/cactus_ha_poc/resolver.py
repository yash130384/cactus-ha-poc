"""Entity Resolver: Translates colloquial German names and aliases to Home Assistant entity_ids."""

import re
import unicodedata
from typing import Optional


class EntityResolver:
    """Robust resolver mapping natural language terms to Home Assistant entity IDs."""

    DEFAULT_LIGHT = "light.esstisch"
    DEFAULT_WEATHER = "weather.forecast_home"

    # Known lighting entities
    LIGHT_REGISTRY = {
        "light.esstisch": [
            "esstisch",
            "esstischlicht",
            "esstischlampe",
            "esstischleuchte",
            "licht am esstisch",
            "lampe am esstisch",
            "leuchte am esstisch",
            "licht beim esstisch",
            "esszimmertisch",
            "esszimmer",
        ],
        "light.kuche": [
            "kuche",
            "kueche",
            "kuechenlicht",
            "kuechenlampe",
            "kuechenleuchte",
            "licht in der kueche",
            "lampe in der kueche",
            "kuechentisch",
        ],
        "light.flur_oben": [
            "flur oben",
            "fluroben",
            "flur_oben",
            "oberer flur",
            "flur",
            "flurlicht",
            "flurlampe",
            "licht im flur",
        ],
        "light.schranklampe": [
            "schranklampe",
            "schranklicht",
            "schrankleuchte",
            "schrank",
            "licht im schrank",
        ],
        "light.leto_bett": [
            "leto bett",
            "letobett",
            "leto_bett",
            "leto",
            "bettlampe",
            "bettlicht",
        ],
        "light.licht_1": [
            "licht 1",
            "licht1",
            "lampe 1",
            "lampe1",
        ],
    }

    # Known weather entities and keywords
    WEATHER_REGISTRY = {
        "weather.forecast_home": [
            "norderstedt",
            "home",
            "zuhause",
            "lokal",
            "hier",
            "draussen",
            "draussen",
            "wetter",
            "regen",
            "regenschirm",
            "wind",
            "temperatur",
            "vorhersage",
        ]
    }

    def __init__(self) -> None:
        # Pre-build lookup dictionaries with normalized keys
        self._light_lookup: dict[str, str] = {}
        for entity_id, aliases in self.LIGHT_REGISTRY.items():
            self._light_lookup[self._normalize(entity_id)] = entity_id
            for alias in aliases:
                self._light_lookup[self._normalize(alias)] = entity_id

        self._weather_lookup: dict[str, str] = {}
        for entity_id, aliases in self.WEATHER_REGISTRY.items():
            self._weather_lookup[self._normalize(entity_id)] = entity_id
            for alias in aliases:
                self._weather_lookup[self._normalize(alias)] = entity_id

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize German strings: lowercase, trim, replace umlauts, strip extra characters."""
        if not text:
            return ""
        s = text.lower().strip()
        # German umlaut replacements
        s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        # Strip non-alphanumeric chars except dots and underscores
        s = re.sub(r"[^\w\s\._-]", " ", s)
        s = re.sub(r"[\s_-]+", " ", s).strip()
        return s

    def resolve_light(self, name: str) -> Optional[str]:
        """Resolve a natural language lamp/room name to a light entity_id."""
        if not name:
            return None

        norm = self._normalize(name)
        if not norm:
            return None

        # Direct match in lookup
        if norm in self._light_lookup:
            return self._light_lookup[norm]

        # Strip generic light suffixes/prefixes
        cleaned = re.sub(r"\b(licht|lampe|leuchte|an|aus)\b", "", norm).strip()
        if cleaned in self._light_lookup:
            return self._light_lookup[cleaned]

        # Substring / keyword search
        for key, entity_id in self._light_lookup.items():
            if len(key) >= 3 and (key in norm or norm in key):
                return entity_id

        # If user simply said "licht" or "lampe" with no other context, fallback to primary test light
        if norm in ("licht", "lampe", "leuchte"):
            return self.DEFAULT_LIGHT

        return None

    def resolve_weather(self, location: str = "") -> str:
        """Resolve a weather location or prompt to a weather entity_id."""
        if not location:
            return self.DEFAULT_WEATHER

        norm = self._normalize(location)
        if not norm:
            return self.DEFAULT_WEATHER

        # Check if matched in weather lookup
        if norm in self._weather_lookup:
            return self._weather_lookup[norm]

        # Substring match
        for key, entity_id in self._weather_lookup.items():
            if len(key) >= 3 and (key in norm or norm in key):
                return entity_id

        # Fallback to default weather entity (e.g. for Fallstrick 4: location="Wind" or unknown place)
        return self.DEFAULT_WEATHER

    def resolve(self, name: str, domain: Optional[str] = None) -> Optional[str]:
        """General resolution method dispatching by domain if specified."""
        if domain == "light":
            return self.resolve_light(name)
        elif domain == "weather":
            return self.resolve_weather(name)

        # Autodetect
        res = self.resolve_light(name)
        if res:
            return res
        return self.resolve_weather(name)
