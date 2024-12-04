#!/usr/bin/env bash

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -q -q -q
.venv/bin/python visualization.py
