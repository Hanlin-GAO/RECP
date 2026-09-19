"""Shared locations for the release's code, input data, and saved models."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORECAST_CODE = ROOT / 'code/forecasting'
FLIGHT_PROCESSED = ROOT / 'data/experiments/flight/processed'
SCENE_DATA = ROOT / 'data/scenes/first_frame'
VEHICLE_POWER = ROOT / 'data/experiments/vehicle_power'
