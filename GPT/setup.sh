#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -d venv ] || /usr/bin/python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_sumo.py
