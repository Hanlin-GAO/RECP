"""Paths for the separated release layout; no dependency on the working directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PV_DATA = ROOT / 'data/forecasting/raw/pv'
WIND_DATA = ROOT / 'data/forecasting/raw/wind'
PV_OUTPUT = Path(os.environ.get('RECP_PV_OUTPUT_DIR', ROOT / 'runs/forecasting/pv'))
WIND_OUTPUT = Path(os.environ.get('RECP_WIND_OUTPUT_DIR', ROOT / 'runs/forecasting/wind'))
