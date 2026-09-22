# cactus-ha-poc

> **Lokaler Proof of Concept (POC): Natürliche Sprachsteuerung für Home Assistant mittels Cactus Needle 3**

`cactus-ha-poc` demonstriert die hochperformante, deterministische Steuerung von Home Assistant über das ultrakompakte Tool-Calling-Foundation-Modell **Cactus Needle 3** (Cactus Compute, Apache 2.0).

---

## 🚀 Highlights & Features

- **Ultrakompakt & Autark:** 35,3 MB Modellgewichte (`needle3.cact`), C-Engine (`libneedle.so`, ~537 KB). Keine schweren ML-Frameworks (weder PyTorch noch ONNX Runtime noch JAX für Inferenz erforderlich).
- **Hohe Genauigkeit für Deutsch:** Zuverlässiges Mapping deutscher Sprachkommandos (*"Schalte das Licht am Esstisch an"*, *"Dimme das Esstischlicht auf 40 Prozent"*, *"Wie ist das Wetter in Norderstedt?"*).
- **Robuster Entity Resolver:** Tolerantes Alias-Mapping und Normalisierung (Umlaute, Suffixe wie `-licht`, `-lampe`) zur Umgehung von *Strict Grounding* Restriktionen.
- **Dualer Modus (Live & Mock):** Vollwertiger Live-REST-Client für Home Assistant Core sowie ein integrierter Simulator für Offline-Betrieb und deterministische Unit-/E2E-Tests.
- **Sub-300ms Latenz:** Lokale CPU-Inferenz inklusive Entitätsauflösung und Home Assistant Service-Call in durchschnittlich ~275 ms (auf Intel Core i5-8400T @ 1.7 GHz).

---

## 📐 Architektur

```
                               Nutzer-Prompt
                    ("Dimme den Esstisch auf 40 Prozent")
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                              NeedleAgent                               │
│  - State Management & needle.reset()                                   │
│  - Zweisprachiger System-Prompt                                        │
│  - Inferenz via Cactus Needle 3 C-Engine (libneedle.so)                │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ function_calls & confidence
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Entity Resolver & Alias Matcher                  │
│  - "Esstisch" / "Esstischlicht"  ───►  light.esstisch                  │
│  - "Küche" / "Küchenlampe"       ───►  light.kuche                     │
│  - "Norderstedt" / "Draußen"     ───►  weather.forecast_home           │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ Technische entity_id & Parameter
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Home Assistant REST Client                      │
│  - Live-Modus:  POST /api/services/light/turn_on                       │
│  - Mock-Modus:  In-Memory State Simulator (light.esstisch, weather)    │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
                               Antwort / CLI
     [ERFOLG] Licht 'Esstisch' auf 40% gedimmt (Status: on).
```

---

## 📦 Installation & Setup

Das Projekt nutzt [`uv`](https://github.com/astral-sh/uv) für blitzschnelle Paketverwaltung unter Python 3.13.

### 1. Repository & Virtualenv
```bash
cd /home/cb/Projects/cactus-ha-poc

# Virtuelles Environment erstellen & Abhängigkeiten installieren
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Konfiguration (`.env`)
Kopiere die Vorlage und passe bei Bedarf deine Home Assistant Daten an:
```bash
cp .env.example .env
```

Inhalt der `.env`:
```ini
# Home Assistant REST-Endpunkt
HASS_URL=http://localhost:8123

# Long-Lived Access Token (aus HA: Profil -> Sicherheit -> Langlebige Zugangs-Token)
# Wenn leer, startet automatisch der Mock-Simulator.
HASS_TOKEN=

# Mock-Modus: true erzwingt den Simulator (keine echten Schaltvorgänge)
MOCK_MODE=true

# Needle 3 Telemetrie (0 = deaktiviert für Datenschutz & Offline-Betrieb)
NEEDLE_TELEMETRY=0

# Mindest-Konfidenz (0.0 bis 1.0)
CONFIDENCE_THRESHOLD=0.40
```

---

## 💻 Verwendung (CLI)

Das CLI-Tool `cactus-ha-poc` bietet drei Betriebsmodi:

### 1. Einmal-Befehl
```bash
# Licht schalten
cactus-ha-poc "Schalte das Licht am Esstisch an"

# Dimmen
cactus-ha-poc "Dimme das Esstischlicht auf 40 Prozent"

# Wetter abfragen
cactus-ha-poc "Brauche ich heute einen Regenschirm draußen?"
```

Beispielausgabe:
```text
[ERFOLG] Licht 'Esstisch' auf 40% gedimmt (Status: on).
  Tool-Call:   control_light(name='Esstischlicht', action='dim', brightness=40)
  Ziel-Entity: light.esstisch
  Konfidenz:   99.8%
  Latenz:      278.4 ms
```

### 2. Interaktiver Dialog-Modus (REPL)
Starte die interaktive Konsole ohne Argumente:
```bash
cactus-ha-poc
```
```text
============================================================
       Cactus HA POC - Sprachsteuerung mit Needle 3         
  Modell: Cactus Needle 3 (35MB) | Modus: MOCK (Simulator)
  Tippe 'exit', 'quit' oder 'q' zum Beenden.
============================================================

HA > Schalte das Licht am Esstisch an
[ERFOLG] Licht 'Esstisch' eingeschaltet (Status: on, Helligkeit: 100%).
  Tool-Call:   control_light(name='Esstisch', action='on')
  Ziel-Entity: light.esstisch
  Konfidenz:   96.8%
  Latenz:      267.1 ms

HA > q
Beende Cactus HA POC.
```

### 3. Benchmark-Evaluation (`--eval`)
Führt die 7 standardisierten Testfälle der Testmatrix durch und gibt eine Leistungsübersicht aus:
```bash
cactus-ha-poc --eval
```

---

## 📊 Benchmark-Ergebnisse (Testmatrix)

Ergebnisse gemessen auf lokaler CPU (Intel Core i5-8400T @ 1.70 GHz, Linux x86_64):

| ID | Test-Kategorie | Eingabe-Prompt | Aufgerufenes Tool | Ziel-Entität | Konfidenz | Latenz | Status |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|
| **TC-01** | Licht Einschalten | *"Schalte das Licht am Esstisch an"* | `control_light` | `light.esstisch` | **96.8%** | 267.1 ms | **PASS** |
| **TC-02** | Licht Ausschalten | *"Esstischlicht ausschalten"* | `control_light` | `light.esstisch` | **100.0%** | 257.1 ms | **PASS** |
| **TC-03** | Licht Dimmen | *"Dimme das Esstischlicht auf 40 Prozent"* | `control_light` | `light.esstisch` | **95.6%** | 306.0 ms | **PASS** |
| **TC-04** | Andere Lampe | *"Küche anmachen"* | `control_light` | `light.kuche` | **96.6%** | 266.9 ms | **PASS** |
| **TC-05** | Wetter Allgemein | *"Wie ist das Wetter in Norderstedt?"* | `get_weather` | `weather.forecast_home` | **65.4%** | 249.3 ms | **PASS** |
| **TC-06** | Regenabfrage | *"Regnet es gerade in Norderstedt?"* | `get_weather` | `weather.forecast_home` | **97.9%** | 272.6 ms | **PASS** |
| **TC-07** | Ohne Ortsangabe | *"Brauche ich heute einen Regenschirm draußen?"* | `get_weather` | `weather.forecast_home` | **96.4%** | 317.7 ms | **PASS** |

**Gesamtergebnis:** **7 / 7 bestanden (100%)**, durchschnittliche Konfidenz **92.7%**, mittlere Latenz **276.7 ms**.

---

## 🧪 Tests

Die vollständige Testsuite umfasst Unit-Tests für Client, Resolver und CLI sowie End-to-End-Inferenztests:
```bash
pytest -v
```

Ausgabe:
```text
tests/test_agent_e2e.py .................... [ 29%]
tests/test_cli.py .....                      [ 44%]
tests/test_ha_client.py ..........           [ 73%]
tests/test_resolver.py .........             [100%]

============================== 34 passed in 9.23s ==============================
```

---

## 🛡️ Gelöste Herausforderungen & Best Practices

1. **Strict Grounding:** Needle 3 unterdrückt Tool-Calls, wenn Parameter technische Strings verlangen, die nicht im Prompt vorkommen (z.B. `light.esstisch`). Durch das Akzeptieren natürlicher Bezeichnungen im Schema (`name: str`) und die nachgelagerte Auflösung im `EntityResolver` arbeitet das System hochgradig robust.
2. **Deutsche Verbsemantik:** Ein konsolidiertes Tool `control_light` mit den Aktionen `['on', 'off', 'dim']` verhindert Fehlzuordnungen deutscher Verben (*an*, *aus*, *dimmen*).
3. **Multi-Turn State Bleed:** Automatischer Aufruf von `needle.reset()` vor jedem unabhängigen Nutzerbefehl verhindert, dass Parameter vorheriger Aufrufe vererbt werden.
4. **Wetter-Orts-Fallback:** Automatische Ergänzung des konfigurierten Heimatstandorts (`weather.forecast_home`, Norderstedt), wenn Nutzeranfragen keine explizite Stadt enthalten.
5. **Datenschutz:** Standardmäßige Deaktivierung von Telemetrie über `NEEDLE_TELEMETRY=0`.

---

## 📜 Lizenz

Apache License 2.0.
