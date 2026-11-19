from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Retrieve or create the singleton SQLAlchemy Engine."""
    global _engine
    if _engine is None:
        if not settings.db_url:
            raise RuntimeError(
                "Database URL is not configured. Please set DB_URL or DATABASE_URL in your environment."
            )
        _engine = create_engine(
            settings.db_url,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Retrieve or create the singleton sessionmaker."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _session_factory


class _SessionLocalProxy:
    """Proxy object so SessionLocal() lazily instantiates the engine/session factory."""

    def __call__(self, *args: Any, **kwargs: Any) -> Session:
        factory = get_session_factory()
        return factory(*args, **kwargs)


if settings.db_url:
    _engine = create_engine(
        settings.db_url,
        pool_pre_ping=True,
        echo=False,
    )
    _session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine,
    )

engine: Engine | None = _engine
SessionLocal: sessionmaker[Session] | None = _session_factory


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a database session per request.
    Ensures safe rollback on exception and cleanup upon completion.
    """
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
