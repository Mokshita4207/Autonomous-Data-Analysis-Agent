import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Loads and validates CSV/Excel datasets.
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

        df.columns = df.columns.str.strip()

        if df.columns.duplicated().any():
            duplicates = list(df.columns[df.columns.duplicated()])

            raise ValueError(
                f"Duplicate column names found: {duplicates}"
            )

        self.validate_file(df)

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
