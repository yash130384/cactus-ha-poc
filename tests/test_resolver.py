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


def test_resolve_decke_variants(resolver: EntityResolver):
    for alias in ["decke 1", "decke1", "erste deckenlampe", "decke eins"]:
        assert resolver.resolve_light(alias) == "light.decke1"

    for alias in ["decke 2", "decke2", "zweite deckenlampe", "decke zwei"]:
        assert resolver.resolve_light(alias) == "light.decke2"

    for alias in ["decke 3", "decke3", "dritte deckenlampe", "decke drei"]:
        assert resolver.resolve_light(alias) == "light.decke3"


def test_resolve_bodenlampe_variants(resolver: EntityResolver):
    for alias in ["bodenlampe", "stehlampe", "bodenleuchte"]:
        assert resolver.resolve_light(alias) == "light.bodenlampe"


def test_resolve_schlafzimmer_variants(resolver: EntityResolver):
    for alias in ["schlafzimmer", "schlafzimmerlicht", "lampe im schlafzimmer"]:
        assert resolver.resolve_light(alias) == "light.schlafzimmer"

    for alias in ["schlafzimmer decke", "schlafzimmerdecke"]:
        assert resolver.resolve_light(alias) == "light.schlafzimmer_decke"


def test_resolve_leto_and_led_variants(resolver: EntityResolver):
    for alias in ["leto bett", "letobett", "bettlampe", "bett"]:
        assert resolver.resolve_light(alias) == "light.leto_bett"

    for alias in ["letos led leiste", "led leiste", "led streifen", "led"]:
        assert resolver.resolve_light(alias) == "light.letos_led_leiste"


def test_resolve_numbered_lights(resolver: EntityResolver):
    for alias in ["licht 1", "licht1", "lampe 1"]:
        assert resolver.resolve_light(alias) == "light.licht_1"

    for alias in ["licht 10", "licht10", "lampe 10"]:
        assert resolver.resolve_light(alias) == "light.licht_10"

    for alias in ["licht 11", "licht11", "lampe 11"]:
        assert resolver.resolve_light(alias) == "light.licht_11"


def test_resolve_all_lights(resolver: EntityResolver):
    for alias in ["alle lichter", "alle lampen", "alles", "lichter", "lampen"]:
        assert resolver.resolve_light(alias) == "light.all"


def test_normalize_umlauts(resolver: EntityResolver):
    # ae -> e, oe -> o, ue -> u
    assert resolver._normalize("Küche") == "kuche"
    assert resolver._normalize("kueche") == "kuche"
    assert resolver._normalize("kuche") == "kuche"
    assert resolver._normalize("Möbel") == "mobel"
    assert resolver._normalize("moebel") == "mobel"
    assert resolver._normalize("Käse") == "kese"
    assert resolver._normalize("kaese") == "kese"


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
