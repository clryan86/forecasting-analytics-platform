# Forecasting Analytics Platform

[![CI](https://github.com/clryan86/forecasting-analytics-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/clryan86/forecasting-analytics-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-forecasting-009688) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

Production-minded forecasting service for training, benchmarking, registering and serving time-series models.

## Engineering highlights
- validated CSV time-series ingestion
- seasonal-naive baseline plus gradient-boosted lag model
- lag and rolling-window feature engineering without target leakage
- holdout backtesting with MAE, RMSE, MAPE and sMAPE
- automatic champion selection by MAE
- persisted/versioned model registry
- FastAPI training and inference endpoints
- lightweight operations dashboard
- Docker, pytest, Ruff and GitHub Actions CI

## Quick start
```bash
python -m venv .venv
pip install -r requirements-dev.txt
python scripts/train_demo.py
uvicorn app.main:app --reload
```
Open the dashboard at `http://127.0.0.1:8000` and OpenAPI at `/docs`.

## Endpoints
`GET /healthz` · `GET /api/models` · `POST /api/train` · `POST /api/forecast`

## Architecture
```text
historical CSV -> validation -> holdout backtest -> model comparison
                                                |
                                                v
                                         champion model
                                                |
                                  +-------------+-------------+
                                  v                           v
                              registry                    metrics
                                  |
                                  v
                           FastAPI inference
```

## Why this project matters
Forecasting is more than fitting a model in a notebook. This repository demonstrates the surrounding ML-engineering system: validation, reproducible evaluation, baseline comparison, artifact persistence, API serving, tests and deployment packaging.

**Christopher Ryan** — Python • Data Engineering • Applied AI • Automation

MIT licensed.
