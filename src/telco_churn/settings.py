"""Configuration YAML validée et chemins indépendants du répertoire courant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Environment = Literal["dev", "staging", "prod"]


def find_project_root() -> Path:
    """Chercher pyproject.toml parmi les parents de ce module."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError("Racine introuvable : fournissez project_root à Settings.load().")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lire un dictionnaire YAML et contextualiser les erreurs de lecture."""
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Impossible de lire la configuration {path} : {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"La configuration {path} doit contenir un dictionnaire YAML.")
    return content


class Settings(BaseModel):
    """Paramètres validés ; utiliser load() pour charger les fichiers YAML.

    Les chemins fournis au modèle doivent déjà être absolus. load() résout les
    chemins YAML relativement à project_root, sans vérifier les données elles-mêmes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: Environment
    project_root: Path
    churn_path: Path
    pilot_path: Path
    churn_target_column: str = Field(min_length=1, strict=True)
    pilot_target_column: str = Field(min_length=1, strict=True)
    log_path: Path
    validation_config_path: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    @field_validator("churn_target_column", "pilot_target_column")
    @classmethod
    def require_non_blank_column(cls, value: str) -> str:
        """Refuser un nom de cible vide sans modifier le nom configuré."""
        if not value.strip():
            raise ValueError("Le nom de la colonne cible ne peut pas être vide.")
        return value

    @field_validator(
        "project_root", "churn_path", "pilot_path", "log_path", "validation_config_path"
    )
    @classmethod
    def require_absolute_path(cls, value: Path) -> Path:
        """Garantir que les consommateurs reçoivent des chemins absolus."""
        if not value.is_absolute():
            raise ValueError("Chemin absolu attendu ; utilisez Settings.load().")
        return value

    @classmethod
    def load(
        cls,
        environment: Environment | None = None,
        *,
        project_root: Path | None = None,
    ) -> Settings:
        """Charger base.yaml puis les surcharges de l'environnement.

        Priorité : argument environment, APP_ENV, puis dev. Une racine explicite
        doit être absolue pour ne pas dépendre du répertoire d'exécution.
        """
        active_env = environment if environment is not None else os.getenv("APP_ENV", "dev")
        if active_env not in ("dev", "staging", "prod"):
            raise ValueError(
                f"Environnement invalide : {active_env!r}. Attendu : dev, staging, prod."
            )

        root = Path(project_root) if project_root is not None else find_project_root()
        if not root.is_absolute():
            raise ValueError("project_root doit être un chemin absolu.")
        root = root.resolve()
        config_dir = root / "configs"
        base_path = config_dir / "base.yaml"
        env_path = config_dir / "environments" / f"{active_env}.yaml"
        values = _read_yaml(base_path) | _read_yaml(env_path)
        reserved = {"environment", "project_root"} & values.keys()
        if reserved:
            raise ValueError(f"Clés réservées interdites dans les YAML : {sorted(reserved)}")

        for key in ("churn_path", "pilot_path", "log_path", "validation_config_path"):
            if key not in values:
                continue  # Pydantic signalera le champ manquant.
            value = values[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} doit être une chaîne de chemin non vide.")
            path = Path(value).expanduser()
            values[key] = (path if path.is_absolute() else root / path).resolve()

        try:
            return cls.model_validate({**values, "environment": active_env, "project_root": root})
        except ValidationError as exc:
            raise ValueError(f"Configuration invalide ({base_path}, {env_path}) :\n{exc}") from exc
