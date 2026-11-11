"""Export traçable de candidats, sans écraser une version existante."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def export_candidates(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    catalog: pd.DataFrame,
    output_dir: Path,
    *,
    version: str = "candidate_v1",
) -> Path:
    """Écrire des CSV alignés et un manifeste explicitement non validé pour l'entraînement."""
    if not version or Path(version).name != version or version in (".", ".."):
        raise ValueError("La version doit être un simple nom de dossier.")
    if not features.index.equals(metadata.index) or len(features) != len(metadata):
        raise ValueError("Features et métadonnées doivent avoir le même index dans le même ordre.")
    if features.columns.tolist() != catalog.feature.tolist():
        raise ValueError("Le catalogue ne correspond pas au schéma exporté.")
    if set(features.columns) & set(metadata.columns):
        raise ValueError("Les métadonnées doivent être séparées des features.")
    destination = output_dir / version
    destination.mkdir(parents=True, exist_ok=False)
    features.to_csv(destination / "features.csv", index=False)
    metadata.to_csv(destination / "metadata.csv", index=False)
    catalog.to_csv(destination / "feature_catalog.csv", index=False)
    manifest = {
        "version": version,
        "status": "candidate",
        "training_validated": False,
        "n_rows": len(features),
        "n_features": len(features.columns),
        "feature_columns": features.columns.tolist(),
        "feature_dtypes": features.dtypes.astype(str).to_dict(),
        "metadata_columns": metadata.columns.tolist(),
        "row_alignment": "same row order in both CSV files",
        "unconfirmed_features": catalog.loc[
            catalog.availability.eq("unconfirmed"), "feature"
        ].tolist(),
        "learned_statistics": False,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def save_processed_dataset(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    catalog: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Sauvegarder le dataset transformé et sa traçabilité sans sortie de diagnostic.

    Les colonnes de contexte et la cible sont séparées des features dans le
    manifeste. Chaque fichier est remplacé atomiquement lors d'une réexécution.
    """
    from tempfile import NamedTemporaryFile

    if not features.index.equals(metadata.index) or len(features) != len(metadata):
        raise ValueError("Features et métadonnées doivent avoir le même index dans le même ordre.")
    if features.columns.tolist() != catalog.feature.tolist():
        raise ValueError("Le catalogue ne correspond pas au schéma exporté.")
    if set(features.columns) & set(metadata.columns):
        raise ValueError("Les colonnes de métadonnées et de features doivent être distinctes.")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "churn_clients_processed.csv"
    combined = pd.concat([metadata.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    manifest = {
        "status": "candidate",
        "training_validated": False,
        "n_rows": len(combined),
        "n_features": len(features.columns),
        "feature_columns": features.columns.tolist(),
        "feature_dtypes": features.dtypes.astype(str).to_dict(),
        "metadata_columns": metadata.columns.tolist(),
        "unconfirmed_features": catalog.loc[
            catalog.availability.eq("unconfirmed"), "feature"
        ].tolist(),
        "learned_statistics": False,
    }
    payloads = [
        (destination, combined.to_csv(index=False)),
        (destination.with_suffix(".catalog.csv"), catalog.to_csv(index=False)),
        (
            destination.with_suffix(".manifest.json"),
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        ),
    ]
    for path, content in payloads:
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_dir, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return destination
