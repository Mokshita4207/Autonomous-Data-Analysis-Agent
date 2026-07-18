"""
StatisticalAnalyzer

Performs statistical analysis on numerical dataset columns.

Responsibilities:
- Pearson correlation analysis
- Spearman correlation analysis
- P-value calculation
- Statistical significance testing
- Correlation matrix generation
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr


class StatisticalAnalyzer:

    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level

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

        if len(numeric_columns) < 2:
            raise ValueError(
                "At least two numeric columns are required."
            )

        return numeric_columns

    def pearson(self, x, y):
        clean_data = pd.DataFrame({
            "x": x,
            "y": y
        }).dropna()

        if len(clean_data) < 3:
            raise ValueError(
                "At least three valid observations are required."
            )

        correlation, pvalue = pearsonr(
            clean_data["x"],
            clean_data["y"]
        )

        return {
            "correlation": float(correlation),
            "pvalue": float(pvalue),
            "significant": bool(
                pvalue < self.significance_level
            )
        }

    def spearman(self, x, y):
        clean_data = pd.DataFrame({
            "x": x,
            "y": y
        }).dropna()

        if len(clean_data) < 3:
            raise ValueError(
                "At least three valid observations are required."
            )

        correlation, pvalue = spearmanr(
            clean_data["x"],
            clean_data["y"]
        )

        return {
            "correlation": float(correlation),
            "pvalue": float(pvalue),
            "significant": bool(
                pvalue < self.significance_level
            )
        }

    def calculate_pvalue(
        self,
        x,
        y,
        method="pearson"
    ):
        clean_data = pd.DataFrame({
            "x": x,
            "y": y
        }).dropna()

        if len(clean_data) < 3:
            raise ValueError(
                "At least three valid observations are required."
            )

        if method == "pearson":
            _, pvalue = pearsonr(
                clean_data["x"],
                clean_data["y"]
            )

        elif method == "spearman":
            _, pvalue = spearmanr(
                clean_data["x"],
                clean_data["y"]
            )

        else:
            raise ValueError(
                "Method must be 'pearson' or 'spearman'."
            )

        return float(pvalue)

    def correlation_analysis(
        self,
        df,
        method="pearson"
    ):
        self.validate_dataframe(df)

        numeric_columns = self.get_numeric_columns(df)

        correlation_matrix = pd.DataFrame(
            index=numeric_columns,
            columns=numeric_columns,
            dtype=float
        )

        pvalue_matrix = pd.DataFrame(
            index=numeric_columns,
            columns=numeric_columns,
            dtype=float
        )

        significant_correlations = []

        for index, column_1 in enumerate(
            numeric_columns
        ):

            for column_2 in numeric_columns[
                index + 1:
            ]:

                clean_data = df[
                    [column_1, column_2]
                ].dropna()

                if len(clean_data) < 3:
                    correlation = None
                    pvalue = None

                elif method == "pearson":
                    correlation, pvalue = pearsonr(
                        clean_data[column_1],
                        clean_data[column_2]
                    )

                elif method == "spearman":
                    correlation, pvalue = spearmanr(
                        clean_data[column_1],
                        clean_data[column_2]
                    )

                else:
                    raise ValueError(
                        "Method must be 'pearson' "
                        "or 'spearman'."
                    )

                correlation_matrix.loc[
                    column_1,
                    column_2
                ] = correlation

                correlation_matrix.loc[
                    column_2,
                    column_1
                ] = correlation

                pvalue_matrix.loc[
                    column_1,
                    column_2
                ] = pvalue

                pvalue_matrix.loc[
                    column_2,
                    column_1
                ] = pvalue

                if (
                    correlation is not None
                    and pvalue is not None
                    and pvalue
                    < self.significance_level
                ):
                    significant_correlations.append({
                        "column_1": column_1,
                        "column_2": column_2,
                        "correlation": float(
                            correlation
                        ),
                        "pvalue": float(
                            pvalue
                        ),
                        "significant": True
                    })

            correlation_matrix.loc[
                column_1,
                column_1
            ] = 1.0

            pvalue_matrix.loc[
                column_1,
                column_1
            ] = 0.0

        last_column = numeric_columns[-1]

        correlation_matrix.loc[
            last_column,
            last_column
        ] = 1.0

        pvalue_matrix.loc[
            last_column,
            last_column
        ] = 0.0

        return {
            "correlation":
                correlation_matrix.to_dict(),

            "pvalue":
                pvalue_matrix.to_dict(),

            "significance":
                significant_correlations,

            "method":
                method,

            "significance_level":
                self.significance_level
        }


if __name__ == "__main__":

    data = {
        "sales": [
            100,
            200,
            300,
            400,
            500
        ],

        "profit": [
            20,
            40,
            60,
            80,
            100
        ]
    }

    dataframe = pd.DataFrame(data)

    analyzer = StatisticalAnalyzer()

    result = analyzer.correlation_analysis(
        dataframe
    )

    print(result)