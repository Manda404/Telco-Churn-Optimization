"""Lecture des CSV sans transformation des données métier."""

from typing import Literal

import pandas as pd
from loguru import logger

from telco_churn.settings import Settings


class TelcoDataLoader:
    """Charger les deux sources à partir des chemins validés par Settings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load_raw(self, dataset: Literal["churn", "pilot"] = "churn") -> pd.DataFrame:
        """Lire un CSV avec les conventions pandas et journaliser ses dimensions."""
        paths = {"churn": self._settings.churn_path, "pilot": self._settings.pilot_path}
        if dataset not in paths:
            raise ValueError(f"Dataset inconnu : {dataset!r}. Attendu : churn ou pilot.")
        path = paths[dataset]
        log = logger.bind(pipeline="data.loaders")
        try:
            frame = pd.read_csv(path)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            log.error("Échec de lecture du dataset {} : {}", dataset, path)
            raise
        log.info("Dataset={} path={} rows={} columns={}", dataset, path, *frame.shape)
        return frame
