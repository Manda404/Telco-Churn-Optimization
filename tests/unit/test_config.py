"""Vérifier la portabilité et les erreurs de configuration."""

from pathlib import Path

import pytest
import yaml

from telco_churn.settings import Settings


@pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
def test_paths_do_not_depend_on_working_directory(monkeypatch, tmp_path, environment):
    monkeypatch.setenv("APP_ENV", environment)
    before = Settings.load()
    monkeypatch.chdir(tmp_path)
    after = Settings.load()
    assert after == before
    assert after.environment == environment
    assert after.churn_path == after.project_root / "data/raw/churn_clients.csv"
    assert after.pilot_path == after.project_root / "data/raw/pilote_campagne_2023.csv"
    assert after.log_path == after.project_root / "logs/telco_churn.log"
    assert after.log_level == ("DEBUG" if environment == "dev" else "INFO")


def test_invalid_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ValueError, match="Environnement invalide"):
        Settings.load()


def test_explicit_environment_has_priority(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    assert Settings.load("dev").environment == "dev"


def test_default_environment(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    assert Settings.load().environment == "dev"


def test_overrides_and_configuration_errors(tmp_path):
    config_dir = tmp_path / "configs"
    (config_dir / "environments").mkdir(parents=True)
    base = Settings.load("dev").model_dump(mode="json", exclude={"environment", "project_root"})
    base_path = config_dir / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    env_path = config_dir / "environments/dev.yaml"
    env_path.write_text("churn_path: custom/clients.csv\n", encoding="utf-8")
    assert Settings.load("dev", project_root=tmp_path).churn_path == (
        tmp_path / "custom/clients.csv"
    )
    for invalid_yaml in ("log_level: TRACE", "typo: value", "churn_path: ''", "- item", "key: ["):
        env_path.write_text(invalid_yaml, encoding="utf-8")
        with pytest.raises(ValueError):
            Settings.load("dev", project_root=tmp_path)
    env_path.unlink()
    with pytest.raises(ValueError, match="Impossible de lire"):
        Settings.load("dev", project_root=tmp_path)


def test_reject_relative_root():
    with pytest.raises(ValueError, match="project_root doit être"):
        Settings.load(project_root=Path("."))
