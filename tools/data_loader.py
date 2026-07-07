import pandas as pd
from pathlib import Path


class DataLoader:
    """
    DataLoader class for loading, validating,
    and performing basic dataset inspection.
    """

    def load_file(self, file_path):

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()

        try:

            if extension == ".csv":

                encodings = [
                    "utf-8",
                    "utf-8-sig",
                    "latin-1",
                    "cp1252"
                ]

                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(
                        "Unable to read CSV using supported encodings."
                    )

            elif extension in [".xlsx", ".xls"]:

                df = pd.read_excel(file_path)

            else:
                raise ValueError(
                    "Only CSV and Excel files are supported."
                )

        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded file is empty.")

        except Exception as e:
            raise ValueError(f"Unable to read file: {e}")

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        # Duplicate column check
        if df.columns.duplicated().any():
            duplicates = list(df.columns[df.columns.duplicated()])
            raise ValueError(
                f"Duplicate column names found: {duplicates}"
            )

        # Validate automatically
        self.validate_file(df)

        return df

    def validate_file(self, df):

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Loaded object is not a pandas DataFrame."
            )

        if df.empty:
            raise ValueError("Dataset is empty.")

        if len(df.columns) == 0:
            raise ValueError("Dataset contains no columns.")

        return "Dataset validation successful."

    def get_basic_info(self, df):

        self.validate_file(df)

        info = {

            "Rows": df.shape[0],

            "Columns": df.shape[1],

            "Shape": df.shape,

            "Column Names": list(df.columns),

            "Data Types": df.dtypes.astype(str).to_dict(),

            "Memory Usage (KB)": round(
                df.memory_usage(deep=True).sum() / 1024,
                2
            )

        }

        return info

    def count_missing_values(self, df):

        self.validate_file(df)

        missing = pd.DataFrame({

            "Missing Values": df.isnull().sum(),

            "Missing Percentage": round(
                df.isnull().mean() * 100,
                2
            )

        })

        return missing

    def detect_column_types(self, df):

        self.validate_file(df)

        column_types = {}

        identifier_keywords = [
            "id",
            "code",
            "number",
            "uuid",
            "key",
            "serial"
        ]

        date_keywords = [
            "date",
            "time",
            "dob",
            "birth",
            "join",
            "created",
            "updated"
        ]

        for column in df.columns:

            column_name = column.lower()

            series = df[column]

            unique_ratio = (
                series.nunique(dropna=True) /
                max(len(series), 1)
            )

            # Identifier
            if (
                any(
                    keyword in column_name
                    for keyword in identifier_keywords
                )
                and unique_ratio > 0.90
            ):

                column_types[column] = "Identifier"

            # Boolean
            elif pd.api.types.is_bool_dtype(series):

                column_types[column] = "Boolean"

            # Datetime dtype
            elif pd.api.types.is_datetime64_any_dtype(series):

                column_types[column] = "Date"

            # Date by column name + successful parsing
            elif any(
                keyword in column_name
                for keyword in date_keywords
            ):

                converted = pd.to_datetime(
                    series,
                    errors="coerce"
                )

                if converted.notna().mean() > 0.80:
                    column_types[column] = "Date"
                else:
                    column_types[column] = "Text"

            # Numeric
            elif pd.api.types.is_numeric_dtype(series):

                column_types[column] = "Numeric"

            # Categorical / Text
            else:

                if unique_ratio < 0.30:
                    column_types[column] = "Categorical"
                else:
                    column_types[column] = "Text"

        return column_types
