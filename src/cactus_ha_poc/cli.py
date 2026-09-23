"""Command Line Interface for cactus-ha-poc: Interactive REPL, Single-Run, and Benchmark Eval."""

import argparse
import sys
import time
from typing import Optional

from .agent import NeedleAgent
from .config import Config, load_config
from .ha_client import MockHomeAssistantClient, create_ha_client

# ANSI escape codes for clean terminal styling
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
GRAY = "\033[90m"


def print_banner(mode_str: str) -> None:
    print(f"{CYAN}{BOLD}============================================================{RESET}")
    print(f"{CYAN}{BOLD}       Cactus HA POC - Sprachsteuerung mit Needle 3         {RESET}")
    print(f"{GRAY}  Modell: Cactus Needle 3 (35MB) | Modus: {mode_str}{RESET}")
    print(f"{GRAY}  Tippe 'exit', 'quit' oder 'q' zum Beenden.{RESET}")
    print(f"{CYAN}{BOLD}============================================================{RESET}\n")


def print_result(res) -> None:
    status_color = GREEN if res.success else RED
    status_text = "ERFOLG" if res.success else "FEHLER"

    print(f"[{status_color}{BOLD}{status_text}{RESET}] {res.message}")
    if res.tool_call:
        args_str = ", ".join(f"{k}={v!r}" for k, v in res.tool_call.get("arguments", {}).items())
        print(f"  {GRAY}Tool-Call:   {RESET}{CYAN}{res.tool_call.get('name')}({args_str}){RESET}")
    if res.entity_id:
        print(f"  {GRAY}Ziel-Entity: {RESET}{YELLOW}{res.entity_id}{RESET}")
    print(f"  {GRAY}Konfidenz:   {RESET}{res.confidence:.1%}")
    print(f"  {GRAY}Latenz:      {RESET}{res.latency_ms:.1f} ms\n")


def run_benchmark(agent: NeedleAgent) -> int:
    """Run benchmark against the test matrix from AGY.md Section 8."""
    print(f"\n{BOLD}{CYAN}=== Starte Benchmark-Evaluation (AGY.md Testmatrix TC-01 bis TC-07) ==={RESET}\n")

    test_cases = [
        {
            "id": "TC-01",
            "category": "Licht Einschalten",
            "prompt": "Schalte das Licht am Esstisch an",
            "expected_tool": "control_light",
            "expected_entity": "light.esstisch",
            "expected_action": "on",
            "min_confidence": 0.85,
        },
        {
            "id": "TC-02",
            "category": "Licht Ausschalten",
            "prompt": "Esstischlicht ausschalten",
            "expected_tool": "control_light",
            "expected_entity": "light.esstisch",
            "expected_action": "off",
            "min_confidence": 0.85,
        },
        {
            "id": "TC-03",
            "category": "Licht Dimmen",
            "prompt": "Dimme das Esstischlicht auf 40 Prozent",
            "expected_tool": "control_light",
            "expected_entity": "light.esstisch",
            "expected_action": "dim",
            "min_confidence": 0.40,
        },
        {
            "id": "TC-04",
            "category": "Andere Lampe",
            "prompt": "Küche anmachen",
            "expected_tool": "control_light",
            "expected_entity": "light.kuche",
            "expected_action": "on",
            "min_confidence": 0.40,
        },
        {
            "id": "TC-05",
            "category": "Wetter Allgemein",
            "prompt": "Wie ist das Wetter in Norderstedt?",
            "expected_tool": "get_weather",
            "expected_entity": "weather.forecast_home",
            "expected_action": "get_weather",
            "min_confidence": 0.40,
        },
        {
            "id": "TC-06",
            "category": "Regenabfrage",
            "prompt": "Regnet es gerade in Norderstedt?",
            "expected_tool": "get_weather",
            "expected_entity": "weather.forecast_home",
            "expected_action": "get_weather",
            "min_confidence": 0.40,
        },
        {
            "id": "TC-07",
            "category": "Ohne Ortsangabe",
            "prompt": "Brauche ich heute einen Regenschirm draußen?",
            "expected_tool": "get_weather",
            "expected_entity": "weather.forecast_home",
            "expected_action": "get_weather",
            "min_confidence": 0.40,
        },
    ]

    results = []
    latencies = []
    confidences = []

    # Warmup
    agent.process_prompt("Warmup")

    header = f"| {'ID':<5} | {'Kategorie':<18} | {'Tool-Call':<15} | {'Entity':<22} | {'Konf.':<7} | {'Latenz':<9} | {'Status':<6} |"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    all_passed = True
    for tc in test_cases:
        prompt = tc["prompt"]
        res = agent.process_prompt(prompt)

        latencies.append(res.latency_ms)
        confidences.append(res.confidence)

        tool_ok = res.tool_call and res.tool_call.get("name") == tc["expected_tool"]
        entity_ok = res.entity_id == tc["expected_entity"]
        conf_ok = res.confidence >= tc["min_confidence"]
        status_ok = res.success and tool_ok and entity_ok and conf_ok

        if not status_ok:
            all_passed = False

        status_str = f"{GREEN}PASS{RESET}" if status_ok else f"{RED}FAIL{RESET}"
        tool_name = (res.tool_call.get("name") if res.tool_call else "none")[:15]
        entity_name = (res.entity_id or "none")[:22]

        row = (
            f"| {tc['id']:<5} | {tc['category']:<18} | {tool_name:<15} | "
            f"{entity_name:<22} | {res.confidence:<6.1%} | {res.latency_ms:<6.1f} ms | {status_str} |"
        )
        print(row)
        results.append((tc, res, status_ok))

    print(divider)

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    passed_count = sum(1 for _, _, ok in results if ok)
    total_count = len(results)

    print(f"\n{BOLD}Benchmark-Zusammenfassung:{RESET}")
    print(f"  Bestanden:           {GREEN if all_passed else RED}{passed_count}/{total_count}{RESET}")
    print(f"  Durchschnittl. Konf.: {avg_conf:.1%}")
    print(f"  Durchschnittl. Latenz:{avg_lat:.1f} ms (Min: {min(latencies):.1f} ms, Max: {max(latencies):.1f} ms)")
    print(f"  Latenz-Budget:        {'ERFUELLT (<500 ms)' if max(latencies) < 500 else 'UEBERSCHRITTEN'}")

    return 0 if all_passed else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cactus-ha-poc",
        description="Home Assistant Sprachsteuerung via Cactus Needle 3",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Einmaliger Sprachbefehl (z.B. 'Schalte das Licht am Esstisch an')",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Führt die standardisierte Benchmark-Testmatrix (TC-01 bis TC-07) aus",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Erzwingt den Live-Modus gegen Home Assistant REST API",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Erzwingt den Mock-Modus (simulierte Geräte)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Schwellenwert für Mindest-Konfidenz (z.B. 0.40)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Führt Befehl im Test-Modus aus (kein physischer Schaltbefehl)",
    )

    args = parser.parse_args(argv)

    cfg = load_config()
    if args.live:
        cfg.mock_mode = False
    elif args.mock:
        cfg.mock_mode = True

    if args.threshold is not None:
        cfg.confidence_threshold = args.threshold

    if args.dry_run:
        cfg.dry_run = True

    mode_label = "MOCK (Simulator)" if cfg.mock_mode or not cfg.hass_token else f"LIVE ({cfg.hass_url})"
    if cfg.dry_run:
        mode_label += " [DRY-RUN / TEST-MODUS]"
    agent = NeedleAgent(config=cfg)

    # Mode 1: Benchmark Evaluation
    if args.eval:
        return run_benchmark(agent)

    # Mode 2: Single command execution
    if args.prompt:
        res = agent.process_prompt(args.prompt, dry_run=args.dry_run)
        print_result(res)
        return 0 if res.success else 1

    # Mode 3: Interactive REPL loop
    print_banner(mode_label)

    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}HA > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print(f"{GRAY}Beende Cactus HA POC.{RESET}")
                break

            res = agent.process_prompt(user_input, dry_run=args.dry_run)
            print_result(res)
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GRAY}Abgebrochen.{RESET}")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
