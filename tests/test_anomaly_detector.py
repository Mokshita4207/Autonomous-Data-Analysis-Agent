import pandas as pd
import pytest

from tools.anomaly_detector import (
    AnomalyDetector
)


@pytest.fixture
def sample_dataframe():

    return pd.DataFrame({

        "value": [
            10,
            11,
            12,
            13,
            14,
            100
        ]

    })


def test_detect_zscore(
    sample_dataframe
):

    detector = AnomalyDetector(
        zscore_threshold=1.5
    )

    result = detector.detect_zscore(
        sample_dataframe
    )

    assert result["method"] == "zscore"

    assert result["outlier_count"] > 0

    assert "anomaly_percentage" in result

    assert "outliers" in result


def test_detect_isolation_forest(
    sample_dataframe
):

    detector = AnomalyDetector(
        contamination=0.2
    )

    result = detector.detect_isolation_forest(
        sample_dataframe
    )

    assert (
        result["method"]
        == "isolation_forest"
    )

    assert "outlier_count" in result

    assert "anomaly_percentage" in result

    assert "outliers" in result


def test_flag_outliers_zscore(
    sample_dataframe
):

    detector = AnomalyDetector()

    result = detector.flag_outliers(
        sample_dataframe,
        method="zscore"
    )

    assert result["method"] == "zscore"


def test_flag_outliers_isolation_forest(
    sample_dataframe
):

    detector = AnomalyDetector()

    result = detector.flag_outliers(
        sample_dataframe,
        method="isolation_forest"
    )

    assert (
        result["method"]
        == "isolation_forest"
    )


def test_invalid_method(
    sample_dataframe
):

    detector = AnomalyDetector()

    with pytest.raises(
        ValueError
    ):

        detector.flag_outliers(
            sample_dataframe,
            method="invalid"
        )