"""Unit tests for EntityResolver."""

import pytest
from cactus_ha_poc.resolver import EntityResolver


@pytest.fixture
def resolver() -> EntityResolver:
    return EntityResolver()


def test_resolve_esstisch_variants(resolver: EntityResolver):
    test_cases = [
        "Esstisch",
        "esstisch",
        "Esstischlicht",
        "esstischlampe",
        "esstischleuchte",
        "Licht am Esstisch",
        "Lampe am Esstisch",
        "esstisch-licht",
        "Esszimmertisch",
    ]
    for case in test_cases:
        assert resolver.resolve_light(case) == "light.esstisch", f"Failed for '{case}'"


def test_resolve_kuche_variants(resolver: EntityResolver):
    test_cases = [
        "Küche",
        "kueche",
        "Küchenlicht",
        "kuechenlicht",
        "Küchenlampe",
        "kuechenlampe",
        "Licht in der Küche",
    ]
    for case in test_cases:
        assert resolver.resolve_light(case) == "light.kuche", f"Failed for '{case}'"


def test_resolve_flur_variants(resolver: EntityResolver):
    test_cases = [
        "Flur",
        "Flur oben",
        "fluroben",
        "oberer Flur",
        "flur_oben",
        "Flurlicht",
    ]
    for case in test_cases:
        assert resolver.resolve_light(case) == "light.flur_oben", f"Failed for '{case}'"


def test_resolve_other_lights(resolver: EntityResolver):
    assert resolver.resolve_light("Schranklampe") == "light.schranklampe"
    assert resolver.resolve_light("Schrank") == "light.schranklampe"
    assert resolver.resolve_light("Leto Bett") == "light.leto_bett"
    assert resolver.resolve_light("Licht 1") == "light.licht_1"


def test_resolve_weather_variants(resolver: EntityResolver):
    test_cases = [
        "Norderstedt",
        "norderstedt",
        "Home",
        "zuhause",
        "draußen",
        "draussen",
        "hier",
        "Wetter",
        "Regen",
        "Wind",
        "Regenschirm",
        "",
        "Unbekannte Stadt 123",  # Fallback to default weather
    ]
    for case in test_cases:
        assert resolver.resolve_weather(case) == "weather.forecast_home", f"Failed for '{case}'"


def test_resolve_generic_fallback(resolver: EntityResolver):
    assert resolver.resolve_light("licht") == "light.esstisch"
    assert resolver.resolve_light("lampe") == "light.esstisch"


def test_resolve_direct_entity_id(resolver: EntityResolver):
    assert resolver.resolve_light("light.esstisch") == "light.esstisch"
    assert resolver.resolve_weather("weather.forecast_home") == "weather.forecast_home"


def test_resolve_unknown_light(resolver: EntityResolver):
    assert resolver.resolve_light("irgendein_raum_der_nicht_existiert") is None


def test_general_resolve_dispatch(resolver: EntityResolver):
    assert resolver.resolve("Esstisch", domain="light") == "light.esstisch"
    assert resolver.resolve("Norderstedt", domain="weather") == "weather.forecast_home"
    assert resolver.resolve("Küche") == "light.kuche"
    assert resolver.resolve("Norderstedt") == "weather.forecast_home"
