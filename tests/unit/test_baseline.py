import numpy as np
import pandas as pd
import pytest

from telco_churn.modeling.baseline import build_baselines, evaluate, select_threshold


def test_preprocessing_stays_on_train_and_accepts_unknown_categories():
    train = pd.DataFrame(
        {
            "fee": [1.0, 3.0, np.nan, 5.0],
            "region": pd.Series(["A", "B", pd.NA, "B"], dtype="string"),
        }
    )
    model = build_baselines(train)["logistic_regression"]
    model.fit(train, [0, 1, 0, 1])
    imputer = model.named_steps["preprocessing"].named_transformers_["numeric"]["imputer"]
    before = imputer.statistics_.copy()
    p = model.predict_proba(pd.DataFrame({"fee": [1000.0, np.nan], "region": ["NEW", "A"]}))
    np.testing.assert_array_equal(before, imputer.statistics_)
    assert before[0] == 3
    assert np.isfinite(p).all()


def test_threshold_and_confusion_counts():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.4]
    threshold = select_threshold(y, p)
    assert threshold == 0.3
    metrics = evaluate(y, p, threshold)
    assert metrics["tp"] == 2 and metrics["fp"] == 0
    assert metrics["f1"] == 1
    with pytest.raises(ValueError):
        evaluate(y, [0, 0, 1, 2])


def test_saved_pipeline_reproduces_predictions(tmp_path):
    import joblib

    train = pd.DataFrame(
        {"fee": [1.0, 2.0, 3.0, 4.0], "region": pd.Series(["A", pd.NA, "B", "B"], dtype="string")}
    )
    pipeline = build_baselines(train)["logistic_regression"].fit(train, [0, 0, 1, 1])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(pipeline.predict_proba(train), restored.predict_proba(train))
