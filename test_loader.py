from tools.data_loader import DataLoader
import pandas as pd


def dataset_report(df):

    print("\nDataset validation successful.\n")

    print("========== BASIC INFORMATION ==========\n")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Shape: {df.shape}")
    print(f"Column Names: {list(df.columns)}")

    dtype_dict = {}

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            dtype_dict[column] = str(df[column].dtype)

        else:

            dtype_dict[column] = "str"

    print(f"Data Types: {dtype_dict}")

    memory = round(

        df.memory_usage(
            deep=True
        ).sum() / 1024,

        2

    )

    print(f"Memory Usage (KB): {memory}")

    print("\n========== CURRENT MISSING VALUES ==========\n")

    missing = pd.DataFrame({

        "Missing Values":
            df.isnull().sum(),

        "Missing Percentage":
            round(
                (df.isnull().sum() / len(df)) * 100,
                2
            )

    })

    print(missing)

    print("\n========== COLUMN TYPES ==========\n")

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            print(f"{column}: Numeric")

        elif pd.api.types.is_datetime64_any_dtype(df[column]):

            print(f"{column}: DateTime")

        elif df[column].nunique() <= max(
            20,
            len(df) * 0.2
        ):

            print(f"{column}: Categorical")

        else:

            print(f"{column}: Text")


loader = DataLoader()

df = loader.load_file("data/sample.csv")

dataset_report(df)
