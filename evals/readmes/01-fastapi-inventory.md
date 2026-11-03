# stockroom

![CI](https://github.com/example/stockroom/actions/workflows/ci.yml/badge.svg) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

A small inventory service for a community food bank. Made by Priya Natarajan.

## Stack

- **FastAPI** for the HTTP API, with Pydantic models for validation
- **PostgreSQL** 16, accessed through SQLAlchemy 2.0; migrations with Alembic
- Background imports run on Celery with Redis as the broker

## Running it

```bash
docker compose up --build
```

Tests use pytest: `pytest -q`. GitHub Actions runs them on every push.
