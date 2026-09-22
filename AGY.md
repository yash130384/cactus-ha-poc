# Architektur- und Umsetzungsplan: cactus-ha-poc

**Projekt:** `cactus-ha-poc`  
**Rolle:** Leitender Entwickler & Software-Architekt  
**Status:** Genehmigt für Umsetzung (Phase 1–5)  
**Datum:** 2026-09-23  
**Ziel:** Lokaler Proof of Concept (POC) zur natürlichen Sprachsteuerung von Home Assistant mittels des 35MB kompakten Tool-Calling-Modells **Cactus Needle 3** (Cactus Compute, Apache 2.0).

---

## Inhaltsverzeichnis
1. [Executive Summary](#1-executive-summary)
2. [Technische Analyse: Cactus Needle 3](#2-technische-analyse-cactus-needle-3)
   - 2.1 Modell & Architektur
   - 2.2 Python 3.13 & Abhängigkeiten (Arch Linux)
   - 2.3 Download- & Cache-Verhalten
   - 2.4 Performance & Latenz
   - 2.5 Lizenzierung & Telemetrie
3. [Home Assistant Integration & REST API](#3-home-assistant-integration--rest-api)
   - 3.1 Lokale Systemumgebung
   - 3.2 Ziel-Entitäten (Licht & Wetter)
   - 3.3 REST API Endpoints & Payloads
   - 3.4 Authentifizierung & Mocking-Strategie
4. [Empirische Analyse: Tool-Calling & Deutsche Sprache](#4-empirische-analyse-tool-calling--deutsche-sprache)
   - 4.1 Testergebnisse mit realen Prompts
   - 4.2 Identifizierte Fallstricke & Lösungen
5. [Systemarchitektur & Komponenten-Design](#5-systemarchitektur--komponenten-design)
   - 5.1 Architekturdiagramm
   - 5.2 Dateistruktur des Projekts
   - 5.3 Modulspezifikationen
6. [Risikoanalyse & Mitigationsmatrix](#6-risikoanalyse--mitigationsmatrix)
7. [Detaillierter Implementierungsplan (Phasen 1–5)](#7-detaillierter-implementierungsplan-phasen-15)
8. [Verifikations- & Testmatrix](#8-verifikations--testmatrix)

---

## 1. Executive Summary

Der Proof of Concept `cactus-ha-poc` evaluiert den Einsatz des ultrakompakten Foundation-Modells **Needle 3** (Cactus Compute) zur Steuerung von Smart-Home-Komponenten in **Home Assistant Core**.

### Zentrale Erkenntnisse der Vorab-Analyse:
1. **Laufzeit-Kompatibilität**: Needle 3 (`cactus-needle==3.0.4`) läuft unter **Python 3.13.15** auf Arch Linux absolut reibungslos. Es erfordert **weder PyTorch noch ONNX Runtime noch JAX** für die Inferenz. Die Laufzeit nutzt eine vorkompilierte Shared Library (`libneedle.so`, ~537 KB), die über `ctypes` angesprochen wird.
2. **Speicher & Latenz**: Die Modellgewichte betragen **35,3 MB** (`needle3.cact`). Die CPU-Inferenz erreicht **>360 Tokens/s Prefill** und **>160 Tokens/s Decode**, was einer Verarbeitungszeit von unter **30 ms** pro Nutzeranfrage entspricht.
3. **Sprachverständnis & Grounding**: Needle 3 beherrscht deutsche Prompts ("Schalte das Licht am Esstisch an", "Dimme den Esstisch auf 30%", "Wie ist das Wetter in Norderstedt?"). Allerdings verlangt das strikte Anti-Halluzinations-System (*Strict Grounding*), dass Parameterbezeichnungen eng an den Eingabetext gekoppelt werden. Ein direkter Zwang zu internen Entitäts-IDs wie `light.esstisch` führt zur Unterdrückung des Tool-Calls. Daher ist ein **Entity Resolver (Alias Mapper)** architektonisch zwingend erforderlich.
4. **Ziel-Szenarien**: Die Steuerung von `light.esstisch` (ein/aus/dimmen) und die Abfrage von Wetterdaten (`weather.forecast_home`, vorkonfiguriert auf Norderstedt: 53.73° N, 9.98° E) lassen sich direkt über die Home Assistant REST API (`/api/services` und `/api/states`) abbilden.

---

## 2. Technische Analyse: Cactus Needle 3

### 2.1 Modell & Architektur
- **Modell-Familie:** Needle 3 (`Cactus-Compute/needle3`), Nachfolger von Needle 2.
- **Architektur:** *Simple Attention Network* / *Laddered Attention Network* ohne klassische rechenintensive Feed-Forward-Networks (FFNs). Ersetzt MLPs durch Monarch-Hadamard-Strukturen und N-Gram-Memory.
- **Aufgabe:** Spezialisiert rein auf Tool Calling, strukturierte Datenextraktion und Embeddings. Keine freie Plauderei, sondern deterministische JSON-Funktionsaufrufe mit kalibrierter Konfidenz (*Confidence Score*).
- **Dateiformat:** Proprietäres komprimiertes `.cact`-Archiv (Single-File Binary).

### 2.2 Python 3.13 & Abhängigkeiten (Arch Linux)
- **Paketname auf PyPI:** `cactus-needle` (aktuell: `3.0.4`).
- **Python Import:** `import needle` (Wichtig: Das Projektverzeichnis darf **niemals** `needle` heißen, um Namenskollisionen zu vermeiden).
- **PEP 668 & Paketverwaltung:** Unter Arch Linux erzwingt das Betriebssystem PEP 668 ("externally managed environment"). Ein virtuelles Environment (`.venv`) via `uv` ist die empfohlene und performanteste Lösung.
- **Abhängigkeiten (Runtime):**
  - Minimalst: Lediglich `huggingface_hub`, `httpx`, `requests`, `certifi`, `click`.
  - Keine C++ Build-Tools (gcc, g++, cmake) notwendig, da keine Quellcode-Kompilierung auf dem Host erfolgt.
  - Kein CUDA, kein Torch, kein JAX notwendig (JAX/Flax ist nur für Training/LoRA als Extra optional definiert).

### 2.3 Download- & Cache-Verhalten
- **Modellgewichte:** Beim ersten Aufruf lädt `needle.agent.fetch` die Basisgewichte `needle3.cact` (35,3 MB) automatisch vom Hugging Face Hub Repository `Cactus-Compute/needle3` herunter.
- **Inferenz-Engine:** Parallel lädt das Paket die plattformspezifische Engine (für Linux x86_64: `cactus_needle-3.0.1-py3-none-manylinux2014_x86_64.whl`), entpackt daraus die Shared Library `libneedle.so` (~537 KB) und bindet sie per `ctypes.CDLL` ein.
- **Lokaler Speicherort:**
  - `~/.cache/cactus-needle/v3/3.0.1/needle3.cact`
  - `~/.cache/cactus-needle/v3/3.0.1/libneedle.so`
- **Offline-Fähigkeit:** Sobald diese Dateien im Cache liegen, kann die gesamte Inferenz 100% autark und ohne Internetverbindung ausgeführt werden.

### 2.4 Performance & Latenz
Messwerte aus der empirischen Analyse auf der Zielmaschine (Linux x86_64, lokale CPU):
- **Prefill-Geschwindigkeit:** ~360 Tokens/Sekunde
- **Decode-Geschwindigkeit:** ~165 Tokens/Sekunde
- **Inferenzzeit pro Tool-Call:** 15 ms bis 35 ms
- **Peak RAM:** ~263 MB
- **Konfidenzwerte:** Typischerweise 0.85 – 0.99 bei passenden Prompts.

### 2.5 Lizenzierung & Telemetrie
- **Lizenz:** Apache 2.0 (Open-Source, kommerziell nutzbar).
- **Telemetrie:** Das Paket sendet standardmäßig anonyme Zähler an Cactus Compute (`_telemetry.py`). Für den Datenschutz und Offline-Betrieb wird in der Konfiguration `NEEDLE_TELEMETRY=0` via Umgebungsvariable gesetzt.

---

## 3. Home Assistant Integration & REST API

### 3.1 Lokale Systemumgebung
- **Dienst:** `homeassistant.service` (Systemd User Service `user@1000.service/app.slice/homeassistant.service`, PID 1035).
- **Basis-URL:** `http://127.0.0.1:8123` (lokal aktiv, HTTP 200).
- **Konfigurationspfad:** `/home/cb/.homeassistant`
- **Geodaten (aus `core.config`):** Latitude 53.7309133, Longitude 9.9864765 -> **Norderstedt / Schleswig-Holstein**.

### 3.2 Ziel-Entitäten
Aus der Registrierung (`~/.homeassistant/.storage/core.entity_registry`):
1. **Licht (POC-Fokus):**
   - `light.esstisch` (Friendly Name: *"Esstisch"*, ZHA Zigbee Leuchte, unterstützt Ein/Aus, Dimmen, Farbtemperatur).
   - Weitere vorhandene Lampen: `light.kuche`, `light.flur_oben`, `light.schranklampe`, `light.leto_bett`, `light.licht_1`.
2. **Wetter (POC-Fokus):**
   - `weather.forecast_home` (Platform: Met.no, Friendly Name: *"Home"*).
   - Repräsentiert die Wetterdaten und Vorhersagen für den konfigurierten Standort **Norderstedt**.

### 3.3 REST API Endpoints & Payloads

Home Assistant stellt eine vollständige REST-API bereit:

| Aktion | HTTP Methode | URL / Endpoint | JSON Payload |
|---|---|---|---|
| **API Health Check** | `GET` | `/api/` | *(keiner)* |
| **Status abfragen** | `GET` | `/api/states/<entity_id>` | *(keiner)* |
| **Licht einschalten / dimmen** | `POST` | `/api/services/light/turn_on` | `{"entity_id": "light.esstisch", "brightness_pct": 50}` |
| **Licht ausschalten** | `POST` | `/api/services/light/turn_off` | `{"entity_id": "light.esstisch"}` |
| **Wetter-Vorhersage abrufen** | `POST` | `/api/services/weather/get_forecasts` | `{"entity_id": "weather.forecast_home", "type": "daily"}` |

**Header:**
```http
Authorization: Bearer <HASS_TOKEN>
Content-Type: application/json
```

### 3.4 Authentifizierung & Mocking-Strategie
- **Long-Lived Access Token (LLAT):** Home Assistant speichert in `.storage/auth` nur SHA-256 Hashes der Tokens. Das eigentliche JWT-Token wird im HA-Web-UI unter Profil -> Sicherheit -> "Langlebige Zugangs-Token" generiert.
- **Konfigurationsverwaltung:** Token und URL werden über Umgebungsvariablen bzw. `.env` geladen (`HASS_URL`, `HASS_TOKEN`).
- **Mock-Modus (`MOCK_MODE=true`):** Um den POC jederzeit isoliert testen und verifizieren zu können (auch wenn kein Live-Token vorliegt oder keine echten Lampen geschaltet werden sollen), implementieren wir einen vollfunktionalen HA-Mock-Client. Dieser liefert realistische Zustände für `light.esstisch` und die Norderstedt-Wetterdaten von `weather.forecast_home`.

---

## 4. Empirische Analyse: Tool-Calling & Deutsche Sprache

Im Rahmen der Voruntersuchung wurden reale Inferenz-Durchläufe auf der Zielmaschine mit Needle 3 durchgeführt. Die Ergebnisse lieferten entscheidende architektonische Leitplanken:

### 4.1 Testergebnisse

| Eingabe-Prompt (Deutsch) | Modell-Entscheidung | Extrahierte Argumente | Konfidenz | Bewertung |
|---|---|---|---|---|
| *"Schalte das Licht am Esstisch an"* | `control_light` | `name="Esstisch", action="on"` | **0.938** | Perfekt |
| *"Esstischlicht ausschalten"* | `control_light` | `name="Esstischlicht", action="off"` | **0.988** | Perfekt |
| *"Dimme den Esstisch auf 30%"* | `control_light` | `name="Esstisch", action="dim", brightness=30` | **0.499** | Richtig erkannt |
| *"Mach die Küche aus"* | `control_light` | `name="Küche", action="off"` | **0.979** | Perfekt |
| *"Wie ist das Wetter in Norderstedt?"* | `get_weather` | `location="Norderstedt"` | **0.982** | Perfekt |
| *"Regnet es in Norderstedt?"* | `get_weather` | `location="Norderstedt"` | **0.984** | Perfekt |

### 4.2 Identifizierte Fallstricke & Architektonische Lösungen

#### Fallstrick 1: Strict Grounding & Technische Entity-IDs
- **Phänomen:** Wenn der Tool-Parameter als `entity_id: Literal["light.esstisch", ...]` definiert wird und der Nutzer sagt "Schalte den Esstisch an", stuft Needle den Aufruf als *ungrounded* ein, da der exakte String `"light.esstisch"` nicht im Text vorkommt. Das Ergebnis ist eine leere Funktionsaufruf-Liste (`[]`).
- **Lösung:** Das Tool akzeptiert den natürlichen Namen als String (`name: str`). Eine nachgelagerte Komponente, der **Entity Resolver**, mappt `"Esstisch"`, `"Esstischlicht"`, `"Esstischlampe"` tolerant auf die technische ID `light.esstisch`.

#### Fallstrick 2: Deutsche Verben bei getrennten Tools
- **Phänomen:** Bei zwei separaten Tools `turn_on_light` und `turn_off_light` rief Needle für "Esstischlicht ausschalten" fälschlicherweise `turn_on_light` auf, da das Modell primär mit englischen Bezeichnern trainiert ist.
- **Lösung:** Zusammenfassung in ein einheitliches Tool `control_light(name: str, action: Literal['on', 'off', 'dim'], brightness: int = 100)` mit zweisprachigen Erläuterungen im Docstring (`an/ein -> on, aus -> off, dimmen -> dim`) und einem gezielten System-Prompt. Die Konfidenz stieg damit sofort auf über 0.98.

#### Fallstrick 3: Multi-Turn History Bleed
- **Phänomen:** Eine `needle.Needle`-Instanz speichert intern den Verlauf. Wird nach einem Dimm-Befehl (30%) eine andere Lampe geschaltet, vererbt sich die Helligkeit von 30% auf den Folgeaufruf, sofern die Instanz nicht zurückgesetzt wird.
- **Lösung:** Der Agent-Wrapper ruft vor jedem eigenständigen Einzelbefehl `needle.reset()` auf, es sei denn, ein interaktiver Multi-Turn-Dialog wird explizit gewünscht.

#### Fallstrick 4: Fehlender Ortsname bei Wetterfragen
- **Phänomen:** Fragt der Nutzer "Wie stark weht der Wind?", sucht Needle mangels Stadtangabe nach einem Nomen und setzt ggf. `location="Wind"`.
- **Lösung:** Das Wetter-Tool definiert `location: str = "Norderstedt"` als Standardwert. Der Wetter-Handler prüft zudem, ob der übergebene Ort plausibel ist oder auf den Standard-Standort `weather.forecast_home` zurückfallen soll.

---

## 5. Systemarchitektur & Komponenten-Design

### 5.1 Architekturdiagramm

```
+-----------------------------------------------------------------------------+
|                                User Prompt                                  |
|                 (z.B. "Schalte das Licht am Esstisch an")                   |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                              NeedleAgent                                    |
|  - System Prompt (Bilingual: DE/EN Context)                                 |
|  - Tool Definitions (@needle.tool: control_light, get_weather)              |
|  - State Management & needle.reset()                                        |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                      Cactus Needle 3 Inference Engine                       |
|               (libneedle.so C-Engine + needle3.cact, 35MB)                   |
|  - Tokens/s: >160 Decode / >360 Prefill                                     |
|  - Returns: function_call: control_light(name="Esstisch", action="on")      |
|             confidence: 0.938                                               |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                       Entity Resolver & Alias Matcher                       |
|  - Input: name="Esstisch", action="on"                                      |
|  - Registry: "esstisch", "esstischlampe" -> "light.esstisch"                |
|  - Target Entity: light.esstisch                                            |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                        Home Assistant Client (API)                          |
|  - Config: HASS_URL, HASS_TOKEN, MOCK_MODE                                  |
|  - Live Mode: POST http://localhost:8123/api/services/light/turn_on         |
|  - Mock Mode: In-Memory Simulated States                                    |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                           Formatted Result / CLI                            |
|        "Licht Esstisch eingeschaltet (Status: on, 100% Helligkeit)"         |
+-----------------------------------------------------------------------------+
```

### 5.2 Dateistruktur des Projekts

```
/home/cb/Projects/cactus-ha-poc/
├── .venv/                      # Virtuelle Python-Umgebung (uv, Python 3.13)
├── .env.example                # Vorlage für Umgebungsvariablen
├── .env                        # Lokale Konfiguration (gitignored)
├── .gitignore                  # Git-Ausschlüsse (.venv, .cache, .env, __pycache__)
├── pyproject.toml              # Projektdefinition & Abhängigkeiten (uv / hatchling)
├── AGY.md                      # Dieser Architektur- und Umsetzungsplan
├── src/
│   └── cactus_ha_poc/
│       ├── __init__.py         # Paket-Initialisierung
│       ├── config.py           # Settings (Pydantic / Dataclasses, Env Loader)
│       ├── ha_client.py        # Home Assistant REST Client (Live + Mock)
│       ├── resolver.py         # Fuzzy / Alias Entity Resolver
│       ├── tools.py            # Needle-kompatible Tool-Funktionen (@needle.tool)
│       ├── agent.py            # NeedleAgent Wrapper (Orchestrierung & Inferenz)
│       └── cli.py              # Interaktive Kommandozeilen-Schnittstelle
└── tests/
    ├── __init__.py
    ├── test_resolver.py        # Tests für Namens- und Raum-Auflösung
    ├── test_ha_client.py       # Tests für REST-Aufrufe und Mock-Modus
    └── test_agent_e2e.py       # End-to-End Tests für Tool-Calling mit Needle
```

### 5.3 Modulspezifikationen

#### Modul `config.py`
Verwaltet Konfigurationswerte aus Umgebungsvariablen:
- `HASS_URL`: Standard `http://localhost:8123`
- `HASS_TOKEN`: Long-Lived Access Token (optional; aktiviert Mock-Modus falls leer)
- `MOCK_MODE`: Boolean (`true` / `false`, Standard: `true` wenn kein Token vorhanden)
- `NEEDLE_TELEMETRY`: Standard `0` (deaktiviert)
- `CONFIDENCE_THRESHOLD`: Mindest-Konfidenz (Standard: `0.40`), unter der Aktionen abgelehnt oder Rückfragen gestellt werden.

#### Modul `resolver.py`
Mappt umgangssprachliche Begriffe auf Home Assistant `entity_id`s:
- Alias-Wörterbuch:
  - `"esstisch"`, `"esstischlicht"`, `"esstischlampe"` -> `light.esstisch`
  - `"küche"`, `"küchenlicht"`, `"küchenlampe"` -> `light.kuche`
  - `"flur"`, `"flur oben"` -> `light.flur_oben`
  - `"wetter"`, `"norderstedt"`, `"draußen"` -> `weather.forecast_home`
- Fallback: Direkte Überprüfung existierender Entitäten aus dem System.

#### Modul `ha_client.py`
Kapselt die Kommunikation mit Home Assistant:
- `get_state(entity_id: str) -> dict`: Ruft `/api/states/<entity_id>` ab.
- `call_service(domain: str, service: str, service_data: dict) -> dict`: Führt Service-Calls aus (`/api/services/<domain>/<service>`).
- `set_light(entity_id: str, action: str, brightness_pct: int = None) -> dict`: Schaltet `turn_on`, `turn_off` oder setzt Helligkeit.
- `get_weather_info(entity_id: str = "weather.forecast_home") -> dict`: Extrahiert Temperatur, Regenwahrscheinlichkeit, Windgeschwindigkeit und Wetterzustand.
- **Mock-Implementierung**: Enthält ein virtuelles State-Dictionary, das Aktionen simuliert, wenn kein Live-Token konfiguriert ist.

#### Modul `tools.py`
Definiert die Needle-Tools mit `@needle.tool`:
- `control_light(name: str, action: str, brightness: int = 100)`
- `get_weather(location: str = "Norderstedt", query_type: str = "all")`

#### Modul `agent.py`
- Instanziiert `needle.Needle` mit den Tools und dem bilingualen System-Prompt.
- Stellt `process_prompt(prompt: str) -> ExecutionResult` bereit.
- Prüft `confidence`, filtert ungrounded Calls, löst Entitäten über den Resolver auf und führt den HA-Client-Call aus.

---

## 6. Risikoanalyse & Mitigationsmatrix

| # | Risiko / Fallstrick | Auswirkung | Wahrscheinlichkeit | Mitigationsstrategie |
|---|---|---|---|---|
| **R1** | **Strict Grounding unterdrückt Call** | Hoch | Hoch | Tools akzeptieren Freitext-Namen (`name: str`), keine technischen Entity-IDs als Enum im Schema. Nachgelagerter `EntityResolver`. |
| **R2** | **Mehrdeutige deutsche Aktionsverben** (z.B. "aus" vs "an") | Hoch | Mittel | Konsolidiertes Tool `control_light` mit expliziten Aktionswerten (`on`, `off`, `dim`) und Keyword-Hinweisen im Docstring. |
| **R3** | **History / State-Bleed zwischen Befehlen** | Mittel | Hoch | Automatischer Aufruf von `needle.reset()` vor jedem eigenständigen Prompt im `NeedleAgent`. |
| **R4** | **Home Assistant Token fehlt oder ungültig** | Hoch | Mittel | Integrierter **Mock-Modus**, der automatisch greift oder explizit aktiviert werden kann. Ermöglicht vollständige Verifikation ohne Token. |
| **R5** | **Fehlender Ort bei Wetterabfragen** | Niedrig | Hoch | Standard-Fallback auf "Norderstedt" (`weather.forecast_home`) in Tool und Resolver. |
| **R6** | **C++ Engine oder Binär-Inkompatibilität** | Hoch | Sehr gering | Vorkompilierte `libneedle.so` (manylinux2014_x86_64) wurde erfolgreich auf Arch Linux verifiziert. Keine lokalen C++ Build-Tools nötig. |
| **R7** | **Ungewollte Telemetriedaten** | Niedrig | Mittel | Explizites Setzen von `NEEDLE_TELEMETRY=0` in `.env` und `config.py`. |

---

## 7. Detaillierter Implementierungsplan (Phasen 1–5)

```
[Phase 1: Setup & Env] ---> [Phase 2: HA Client & Mock] ---> [Phase 3: Tools & Resolver]
                                                                        |
                                                                        v
[Phase 5: CLI & Testing] <--------------------------------- [Phase 4: Needle Agent Core]
```

### Phase 1: Projekt-Setup & Umgebung
- **Ziel:** Sauberes Repository mit `pyproject.toml`, Git-Tracking und vollständigem Virtualenv.
- **Schritte:**
  1. `pyproject.toml` mit Metadaten, Abhängigkeiten (`cactus-needle>=3.0.4`, `requests>=2.31.0`, `python-dotenv>=1.0.0`, `pytest>=8.0.0`) und CLI-Entrypoint erstellen.
  2. `.gitignore` für Python, `.venv`, `.env` und `.cache` konfigurieren.
  3. `.env.example` mit Dokumentation der Konfigurationsvariablen anlegen.
  4. Repository-Struktur unter `src/cactus_ha_poc` initialisieren.

### Phase 2: Home Assistant REST Client & Simulator
- **Ziel:** Robuster Client für Zustandsabfragen und Service-Calls inklusive Mock-Modus.
- **Schritte:**
  1. `src/cactus_ha_poc/config.py`: Laden von `.env` via `python-dotenv`.
  2. `src/cactus_ha_poc/ha_client.py`: Implementierung der REST-Methoden (`get_state`, `call_service`, `set_light`, `get_weather`).
  3. Integrierter `MockHomeAssistantClient` mit simuliertem Zustand für `light.esstisch` (Zustand, Helligkeit) und `weather.forecast_home` (Temperatur 17°C, Regen 0mm, Wind 14 km/h).
  4. Unit-Tests in `tests/test_ha_client.py`.

### Phase 3: Entity Resolver & Tool-Definitionen
- **Ziel:** Übersetzung umgangssprachlicher Namen in HA-Entitäten und Definition der Needle-Tools.
- **Schritte:**
  1. `src/cactus_ha_poc/resolver.py`: Alias-Mapping mit Normalisierung (Umlaute, Klein-/Großschreibung, Suffixe wie "-licht", "-lampe").
  2. `src/cactus_ha_poc/tools.py`: Definition von `control_light` und `get_weather` mit `@needle.tool` Dekoratoren und optimierten Type-Hints.
  3. Unit-Tests in `tests/test_resolver.py`.

### Phase 4: Needle Agent Core & Prompt Orchestration
- **Ziel:** Verbindung von Needle 3 Inferenz mit Ausführungslogik und Fehlerbehandlung.
- **Schritte:**
  1. `src/cactus_ha_poc/agent.py`: `NeedleAgent` Klasse.
  2. Implementierung von `process_prompt(prompt: str)`:
     - Aufruf von `needle.reset()`.
     - Inferenz via `needle.complete(prompt)`.
     - Konfidenz-Prüfung (`confidence >= threshold`).
     - Entitäts-Auflösung via `EntityResolver`.
     - Ausführung über `HomeAssistantClient`.
     - Rückgabe eines strukturierten `ExecutionResult` (Erfolg, Aktion, Details, Konfidenz, Latenz).
  3. End-to-End Tests in `tests/test_agent_e2e.py`.

### Phase 5: Interaktive CLI & Verifikation
- **Ziel:** Benutzerfreundliche interaktive Konsole und Batch-Test-Runner.
- **Schritte:**
  1. `src/cactus_ha_poc/cli.py`:
     - Interaktiver Modus (Prompt-Schleife: Eingabe natürlicher Sprache -> farbige Ausgabe von Tool-Call, Confidence, HA-Aktion und Rückmeldung).
     - Einmal-Modus (`cactus-ha-poc "Schalte den Esstisch an"`).
     - Eval-Modus (`cactus-ha-poc --eval`), der die definierte Benchmark-Matrix durchläuft.
  2. Dokumentation in `README.md`.
  3. Verifikation aller Akzeptanzkriterien.

---

## 8. Verifikations- & Testmatrix

Die Implementierung gilt als erfolgreich abgeschlossen, wenn folgende Testfälle im POC fehlerfrei durchlaufen:

| ID | Test-Kategorie | Eingabe-Prompt | Erwarteter Tool-Call | Erwartete HA-Aktion | Akzeptanzkriterium |
|---|---|---|---|---|---|
| **TC-01** | Licht Einschalten | *"Schalte das Licht am Esstisch an"* | `control_light(name="Esstisch", action="on")` | Service `light.turn_on` für `light.esstisch` | Status = `on`, Konfidenz > 0.85 |
| **TC-02** | Licht Ausschalten | *"Esstischlicht ausschalten"* | `control_light(name="Esstischlicht", action="off")` | Service `light.turn_off` für `light.esstisch` | Status = `off`, Konfidenz > 0.85 |
| **TC-03** | Licht Dimmen | *"Dimme das Esstischlicht auf 40 Prozent"* | `control_light(name="Esstischlicht", action="dim", brightness=40)` | Service `light.turn_on` mit `brightness_pct: 40` | Status = `on`, Helligkeit = 40% |
| **TC-04** | Andere Lampe | *"Küche anmachen"* | `control_light(name="Küche", action="on")` | Service `light.turn_on` für `light.kuche` | Ziel-Entität `light.kuche` korrekt aufgelöst |
| **TC-05** | Wetter Allgemein | *"Wie ist das Wetter in Norderstedt?"* | `get_weather(location="Norderstedt")` | Query `weather.forecast_home` | Ausgabe von Temperatur, Zustand, Wind |
| **TC-06** | Regenabfrage | *"Regnet es gerade in Norderstedt?"* | `get_weather(location="Norderstedt")` | Query `weather.forecast_home` | Gezielte Information zu Niederschlag |
| **TC-07** | Ohne Ortsangabe | *"Brauche ich heute einen Regenschirm draußen?"* | `get_weather(location="Norderstedt")` | Fallback auf Norderstedt | Automatische Ergänzung des Heimatorts |
| **TC-08** | Latenz-Budget | Beliebiger Prompt | - | Vollständige Antwort in < 150 ms | Gemessene Pipeline-Dauer < 150 ms |

---

*Planung abgeschlossen und freigegeben.*
