import pandas as pd
import pytest

from tools.statistical_analyzer import (
    StatisticalAnalyzer
)


@pytest.fixture
def sample_dataframe():

    return pd.DataFrame({

        "x": [
            1,
            2,
            3,
            4,
            5
        ],

        "y": [
            2,
            4,
            6,
            8,
            10
        ],

        "z": [
            5,
            4,
            3,
            2,
            1
        ]

    })


def test_pearson(
    sample_dataframe
):

    analyzer = StatisticalAnalyzer()

    result = analyzer.pearson(
        sample_dataframe["x"],
        sample_dataframe["y"]
    )

    assert result["correlation"] > 0.99

    assert result["pvalue"] < 0.05

    assert result["significant"] is True


def test_spearman(
    sample_dataframe
):

    analyzer = StatisticalAnalyzer()

    result = analyzer.spearman(
        sample_dataframe["x"],
        sample_dataframe["y"]
    )

    assert result["correlation"] > 0.99

    assert result["significant"] is True


def test_calculate_pvalue(
    sample_dataframe
):

    analyzer = StatisticalAnalyzer()

    pvalue = analyzer.calculate_pvalue(
        sample_dataframe["x"],
        sample_dataframe["y"]
    )

    assert isinstance(
        pvalue,
        float
    )

    assert pvalue < 0.05


def test_correlation_analysis(
    sample_dataframe
):

    analyzer = StatisticalAnalyzer()

    result = analyzer.correlation_analysis(
        sample_dataframe
    )

    assert "correlation" in result

    assert "pvalue" in result

    assert "significance" in result

    assert len(
        result["significance"]
    ) > 0


def test_invalid_dataframe():

    analyzer = StatisticalAnalyzer()

    with pytest.raises(
        TypeError
    ):

        analyzer.correlation_analysis(
            "invalid"
        )