# SupportIQ: Deep Learning Customer Support Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)

SupportIQ is a production-grade AI system designed to automatically understand, classify, and route customer support tickets. It leverages Deep Learning (DistilBERT) to predict ticket categories, detect urgency, analyze sentiment, and generate suggested responses for support agents.

## Features
- **Intelligent Classification**: Fine-tuned DistilBERT model for accurate intent detection across 5 business categories (Billing, Delivery, Refund, Account Access, Technical Issue).
- **Heuristic Triage**: Rule-based urgency and sentiment detection to prioritize high-value or frustrated customers.
- **Production API**: FastAPI backend with Pydantic validation, async processing, and Swagger UI documentation.
- **Data Persistence**: SQLAlchemy ORM integrated with PostgreSQL for auditing predictions and model versions.
- **Containerized Deployment**: Fully dockerized application with `docker-compose` for reproducible, one-command local setup.
- **Automated Testing**: `pytest` integration tests with mocked ML pipelines for fast, reliable CI/CD.

## Tech Stack
- **Machine Learning**: PyTorch, Hugging Face Transformers (`distilbert-base-uncased`), Scikit-Learn (Baseline)
- **Backend**: FastAPI, Pydantic, SQLAlchemy, Uvicorn
- **Database**: PostgreSQL (Production), SQLite (Local Dev fallback)
- **DevOps**: Docker, Docker Compose, GitHub Actions (CI), Pytest

## Model Evaluation
The classification pipeline was evaluated on a stratified 80/20 train/test split of the Bitext Customer Support dataset. 
- **Baseline**: TF-IDF + Logistic Regression (Macro-F1: ~0.85)
- **Production Model**: Fine-tuned DistilBERT (Macro-F1: **~0.94+**)
*Note: Class weights and Macro-F1 were used to ensure robust performance on underrepresented ticket categories.*

## Quick Start

### Docker 
Ensure Docker Desktop is running, then execute:
```bash
docker-compose up --build