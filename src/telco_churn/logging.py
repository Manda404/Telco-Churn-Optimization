"""Configuration centralisée de Loguru, inspirée du projet BNPL."""

import sys

from loguru import logger

from telco_churn.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configurer une fois les sorties console et fichier au démarrage."""
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.configure(extra={"pipeline": "-"})
    template = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[pipeline]} | {message}"
    logger.add(sys.stderr, level=settings.log_level, format=template, diagnose=False)
    logger.add(
        settings.log_path,
        level=settings.log_level,
        format=template,
        encoding="utf-8",
        diagnose=False,
    )
