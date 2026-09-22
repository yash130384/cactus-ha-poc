"""Configuration management for cactus-ha-poc."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from environment or .env file."""
    hass_url: str = "http://localhost:8123"
    hass_token: Optional[str] = None
    mock_mode: bool = True
    needle_telemetry: str = "0"
    confidence_threshold: float = 0.40

    def __post_init__(self) -> None:
        # Enforce telemetry setting in the environment immediately
        os.environ["NEEDLE_TELEMETRY"] = str(self.needle_telemetry)


def load_config(env_path: Optional[str | Path] = None) -> Config:
    """Load configuration from environment and optional .env file.
    
    If no specific env_path is provided, standard .env resolution is used.
    """
    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=False)

    hass_url = os.getenv("HASS_URL", "http://localhost:8123").rstrip("/")
    hass_token = os.getenv("HASS_TOKEN", "").strip() or None

    mock_mode_raw = os.getenv("MOCK_MODE", "").lower().strip()
    if mock_mode_raw in ("true", "1", "yes", "on"):
        mock_mode = True
    elif mock_mode_raw in ("false", "0", "no", "off"):
        mock_mode = False
    else:
        # Default: if no token is provided, fallback to mock mode
        mock_mode = hass_token is None

    needle_telemetry = os.getenv("NEEDLE_TELEMETRY", "0").strip()

    try:
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))
    except ValueError:
        confidence_threshold = 0.40

    cfg = Config(
        hass_url=hass_url,
        hass_token=hass_token,
        mock_mode=mock_mode,
        needle_telemetry=needle_telemetry,
        confidence_threshold=confidence_threshold,
    )
    return cfg
