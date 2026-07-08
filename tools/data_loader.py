import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Loads and validates CSV/Excel datasets.
    """

    def load_file(self, file_path):

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

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

                        df = pd.read_csv(
                            file_path,
                            encoding=encoding
                        )

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

            raise ValueError(
                "The uploaded file is empty."
            )

        except Exception as e:

            raise ValueError(
                f"Unable to read file: {e}"
            )

        df.columns = df.columns.str.strip()

        if df.columns.duplicated().any():

            duplicates = list(
                df.columns[df.columns.duplicated()]
            )

            raise ValueError(
                f"Duplicate column names found: {duplicates}"
            )

        self.validate_file(df)

        df = self.handle_missing_data(df)

        return df

    def validate_file(self, df):

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Loaded object is not a pandas DataFrame."
            )

        if df.empty:

            raise ValueError(
                "Dataset is empty."
            )

        if len(df.columns) == 0:

            raise ValueError(
                "Dataset contains no columns."
            )

        return "Dataset validation successful."

    def handle_missing_data(self, df):

        print("\n========== MISSING VALUES BEFORE HANDLING ==========\n")

        before = pd.DataFrame({

            "Missing Values":
                df.isnull().sum(),

            "Missing Percentage":
                round(
                    (df.isnull().sum() / len(df)) * 100,
                    2
                )

        })

        print(before)

        filled = 0

        for column in df.columns:

            missing = df[column].isnull().sum()

            if missing > 0:

                filled += missing

                if pd.api.types.is_numeric_dtype(df[column]):

                    df[column] = df[column].fillna(
                        df[column].mean()
                    )

                else:

                    mode = df[column].mode()

                    if not mode.empty:

                        df[column] = df[column].fillna(
                            mode[0]
                        )

                    else:

                        df[column] = df[column].fillna(
                            "Unknown"
                        )

        print(f"\nMissing values handled: {filled}")

        print("\n========== MISSING VALUES AFTER HANDLING ==========\n")

        after = pd.DataFrame({

            "Missing Values":
                df.isnull().sum(),

            "Missing Percentage":
                round(
                    (df.isnull().sum() / len(df)) * 100,
                    2
                )

        })

        print(after)

        return df
