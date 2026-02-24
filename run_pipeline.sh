#!/usr/bin/env bash
set -e

# Create and activate a venv, install deps, and run preprocessing
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python preprocessing.py
