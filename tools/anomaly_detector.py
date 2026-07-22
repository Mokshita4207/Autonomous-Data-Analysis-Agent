"""
AnomalyDetector

Detects outliers and anomalies using:
- Z-score
- Isolation Forest
"""

import pandas as pd

from sklearn.ensemble import IsolationForest


class AnomalyDetector:

    def __init__(
        self,
        zscore_threshold=3.0,
        contamination=0.05,
        random_state=42
    ):
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.random_state = random_state

    def validate_dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "DataFrame cannot be empty."
            )

        return True

    def get_numeric_columns(self, df):
        self.validate_dataframe(df)

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if not numeric_columns:
            raise ValueError(
                "No numeric columns available."
            )

        return numeric_columns

    def detect_zscore(
        self,
        df,
        columns=None
    ):
        self.validate_dataframe(df)

        if columns is None:
            columns = self.get_numeric_columns(df)

        if not columns:
            raise ValueError(
                "No numeric columns available."
            )

        numeric_data = df[columns].copy()

        standard_deviation = numeric_data.std(
            ddof=0
        ).replace(0, 1)

        z_scores = (
            numeric_data
            - numeric_data.mean()
        ) / standard_deviation

        outlier_mask = (
            z_scores.abs()
            > self.zscore_threshold
        )

        row_outlier_mask = outlier_mask.any(
            axis=1
        )

        outlier_rows = df[
            row_outlier_mask
        ].copy()

        outlier_rows[
            "outlier_columns"
        ] = (
            outlier_mask[
                row_outlier_mask
            ]
            .apply(
                lambda row:
                row.index[
                    row
                ].tolist(),
                axis=1
            )
        )

        return {
            "method": "zscore",

            "threshold":
                self.zscore_threshold,

            "outlier_count":
                int(
                    row_outlier_mask.sum()
                ),

            "anomaly_percentage":
                float(
                    row_outlier_mask.mean()
                    * 100
                ),

            "outliers":
                outlier_rows.to_dict(
                    orient="records"
                )
        }

    def detect_isolation_forest(
        self,
        df,
        columns=None
    ):
        self.validate_dataframe(df)

        if columns is None:
            columns = self.get_numeric_columns(df)

        if not columns:
            raise ValueError(
                "No numeric columns available."
            )

        numeric_data = df[
            columns
        ].copy()

        numeric_data = numeric_data.fillna(
            numeric_data.median()
        )

        model = IsolationForest(
            contamination=
                self.contamination,

            random_state=
                self.random_state
        )

        predictions = model.fit_predict(
            numeric_data
        )

        anomaly_mask = (
            predictions == -1
        )

        outliers = df[
            anomaly_mask
        ].copy()

        outliers[
            "anomaly_score"
        ] = model.decision_function(
            numeric_data[
                anomaly_mask
            ]
        )

        return {
            "method":
                "isolation_forest",

            "contamination":
                self.contamination,

            "outlier_count":
                int(
                    anomaly_mask.sum()
                ),

            "anomaly_percentage":
                float(
                    anomaly_mask.mean()
                    * 100
                ),

            "outliers":
                outliers.to_dict(
                    orient="records"
                )
        }

    def flag_outliers(
        self,
        df,
        method="zscore",
        columns=None
    ):
        if method == "zscore":
            return self.detect_zscore(
                df,
                columns
            )

        if method == "isolation_forest":
            return self.detect_isolation_forest(
                df,
                columns
            )

        raise ValueError(
            "Method must be 'zscore' "
            "or 'isolation_forest'."
        )


if __name__ == "__main__":

    data = {
        "value": [
            10,
            11,
            12,
            13,
            14,
            100
        ]
    }

    dataframe = pd.DataFrame(data)

    detector = AnomalyDetector()

    result = detector.flag_outliers(
        dataframe,
        method="zscore"
    )

    print(result)