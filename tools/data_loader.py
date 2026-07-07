import pandas as pd
from pathlib import Path


class DataLoader:
    """
    DataLoader class for loading and validating datasets.
    """

    def load_file(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()

        try:
            if extension == ".csv":
                df = pd.read_csv(file_path)

            elif extension in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)

            else:
                raise ValueError("Only CSV and Excel files are supported.")

        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded file is empty.")

        return df

    def validate_file(self, df):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Loaded object is not a pandas DataFrame.")

        if df.empty:
            raise ValueError("Dataset is empty.")

        if len(df.columns) == 0:
            raise ValueError("Dataset contains no columns.")

        return "Dataset validation successful."

    def get_basic_info(self, df):

        info = {
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Shape": df.shape,
            "Column Names": list(df.columns),
            "Data Types": df.dtypes.astype(str).to_dict(),
            "Memory Usage (KB)": round(df.memory_usage(deep=True).sum() / 1024, 2)
        }

        return info

    def count_missing_values(self, df):

        missing = pd.DataFrame({
            "Missing Values": df.isnull().sum(),
            "Missing Percentage": round(df.isnull().mean() * 100, 2)
        })

        return missing

    def detect_column_types(self, df):

        column_types = {}

        for column in df.columns:

            column_name = column.lower()

            # Numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                column_types[column] = "Numeric"

            # Boolean
            elif pd.api.types.is_bool_dtype(df[column]):
                column_types[column] = "Boolean"

            # Identifier
            elif any(keyword in column_name for keyword in ["id", "code", "number"]):
                column_types[column] = "Identifier"

            # Date
            elif any(keyword in column_name for keyword in
                     ["date", "time", "dob", "birth", "join", "created", "updated"]):
                column_types[column] = "Date"

            # Categorical or Text
            else:
                unique_ratio = df[column].nunique() / len(df)

                if unique_ratio < 0.30:
                    column_types[column] = "Categorical"
                else:
                    column_types[column] = "Text"

        return column_types