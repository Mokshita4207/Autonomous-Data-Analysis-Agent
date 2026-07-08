import pandas as pd


class ColumnClassifier:
    """
    Owns all column classification logic:
      - detect_column_types()
      - detect_subtypes()      (Ordinal vs Nominal)
      - detect_confidence()    (High/Medium/Low)
      - detect_role_hint()     (target/grouping/measure/feature/none)
    """

    IDENTIFIER_KEYWORDS = ["id", "code", "number", "uuid", "key", "serial"]
    DATE_KEYWORDS = ["date", "time", "dob", "birth", "join", "created", "updated"]
    TARGET_KEYWORDS = ["target", "label", "outcome", "churn", "status", "result", "class"]
    GROUPING_KEYWORDS = ["region", "segment", "group", "department", "category", "type"]

    # Numeric columns that represent an aggregated/computed business
    # metric -> "measure". Numeric columns that represent a raw
    # per-record attribute (age, height, salary) -> "feature".
    MEASURE_KEYWORDS = ["revenue", "sales", "amount", "price", "cost", "total",
                         "count", "score", "profit", "spend", "sum"]

    ORDINAL_PATTERNS = [
        {"low", "medium", "high"},
        {"low", "med", "high"},
        {"small", "medium", "large"},
        {"small", "med", "large"},
        {"poor", "fair", "good", "excellent"},
        {"bad", "average", "good"},
        {"1", "2", "3", "4", "5"},
        {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
        {"never", "rarely", "sometimes", "often", "always"},
    ]


    CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.05

    def classify(self, df):
        types = self.detect_column_types(df)
        subtypes = self.detect_subtypes(df, types)
        confidences = self.detect_confidence(df, types)
        role_hints = self.detect_role_hint(df, types)

        result = {}
        for column in df.columns:
            result[column] = {
                "type": types[column],
                "subtype": subtypes[column],
                "confidence": confidences[column],
                "role_hint": role_hints[column],
            }
        return result

    def detect_column_types(self, df):
        column_types = {}

        for column in df.columns:
            column_name = column.lower()
            series = df[column]
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

            if (any(k in column_name for k in self.IDENTIFIER_KEYWORDS)
                    and unique_ratio > 0.90):
                column_types[column] = "Identifier"

            elif pd.api.types.is_bool_dtype(series):
                column_types[column] = "Boolean"

            elif pd.api.types.is_datetime64_any_dtype(series):
                column_types[column] = "Date"

            elif any(k in column_name for k in self.DATE_KEYWORDS):
                converted = pd.to_datetime(series, errors="coerce")
                column_types[column] = "Date" if converted.notna().mean() > 0.80 else "Text"

            elif pd.api.types.is_numeric_dtype(series):
                column_types[column] = "Numeric"

            elif (pd.api.types.is_object_dtype(series)
                    or pd.api.types.is_string_dtype(series)):
                column_types[column] = (
                    "Categorical" if unique_ratio <= self.CATEGORICAL_UNIQUE_RATIO_THRESHOLD
                    else "Text"
                )

            else:
                column_types[column] = (
                    "Categorical" if unique_ratio <= self.CATEGORICAL_UNIQUE_RATIO_THRESHOLD
                    else "Text"
                )

        return column_types

    def detect_subtypes(self, df, column_types=None):
        if column_types is None:
            column_types = self.detect_column_types(df)

        subtypes = {}
        for column, col_type in column_types.items():
            if col_type != "Categorical":
                subtypes[column] = None
                continue

            clean = df[column].dropna().astype(str).str.strip().str.lower()
            unique_values = set(clean.unique())

            subtype = "Nominal"
            for pattern in self.ORDINAL_PATTERNS:
                if unique_values and unique_values.issubset(pattern):
                    subtype = "Ordinal"
                    break
            subtypes[column] = subtype

        return subtypes

    def detect_confidence(self, df, column_types=None):
        if column_types is None:
            column_types = self.detect_column_types(df)

        confidences = {}
        for column, col_type in column_types.items():
            series = df[column]
            clean = series.dropna()

            if clean.empty:
                confidences[column] = "Low"
            elif col_type in ("Numeric", "Boolean"):
                confidences[column] = "High"
            elif col_type == "Date":
                confidences[column] = "High" if pd.api.types.is_datetime64_any_dtype(series) else "Medium"
            elif col_type == "Identifier":
                unique_ratio = clean.nunique() / len(clean)
                confidences[column] = "High" if unique_ratio > 0.98 else "Medium"
            elif col_type == "Categorical":
                unique_ratio = clean.nunique() / len(clean)
                confidences[column] = "High" if unique_ratio <= self.CATEGORICAL_UNIQUE_RATIO_THRESHOLD / 2 else "Medium"
            elif col_type == "Text":
                confidences[column] = "Medium"
            else:
                confidences[column] = "Low"

        return confidences

    def detect_role_hint(self, df, column_types=None):
        if column_types is None:
            column_types = self.detect_column_types(df)

        role_hints = {}
        for column, col_type in column_types.items():
            column_name = column.lower()

            if any(k in column_name for k in self.TARGET_KEYWORDS):
                role_hints[column] = "target"
            elif col_type == "Categorical" and any(k in column_name for k in self.GROUPING_KEYWORDS):
                role_hints[column] = "grouping"
            elif col_type == "Numeric":
                # Aggregated business metric -> measure.
                # Raw per-record attribute (age, height, salary) -> feature.
                if any(k in column_name for k in self.MEASURE_KEYWORDS):
                    role_hints[column] = "measure"
                else:
                    role_hints[column] = "feature"
            else:
                role_hints[column] = "none"

        return role_hints