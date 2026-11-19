from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy database models."""
    pass


# Ensure all models are registered on Base.metadata for Alembic migrations
import app.models  # noqa: F401, E402
