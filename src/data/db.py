"""Database access helpers (SQLAlchemy 2.0)."""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from src.utils.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_engine() -> Engine:
    """Process-wide engine for the application database."""
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


def write_df(
    df: pd.DataFrame,
    table: str,
    schema: str,
    engine: Engine | None = None,
    if_exists: str = "replace",
    chunksize: int = 10_000,
) -> int:
    """Write a DataFrame to ``schema.table``; returns row count."""
    engine = engine or get_engine()
    df.to_sql(
        table,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    logger.info("db_write", extra={"table": f"{schema}.{table}", "rows": len(df)})
    return len(df)


def read_df(query: str, engine: Engine | None = None, **params: object) -> pd.DataFrame:
    """Read a SQL query into a DataFrame."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params or None)


def execute(sql: str, engine: Engine | None = None, **params: object) -> None:
    """Execute a DDL/DML statement."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), params or None)


def table_count(schema: str, table: str, engine: Engine | None = None) -> int:
    """Row count of a table (0 if missing)."""
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT count(*) FROM {schema}.{table}"))
            return int(result.scalar() or 0)
    except Exception:  # table absent
        return 0
