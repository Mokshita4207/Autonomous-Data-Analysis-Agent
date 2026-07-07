from tools.data_loader import DataLoader


def main():

    loader = DataLoader()

    try:
        # Load and validate dataset
        df = loader.load_file("data/sample.csv")

        print("Dataset validation successful.")

        print("\n========== BASIC INFORMATION ==========\n")

        info = loader.get_basic_info(df)

        for key, value in info.items():
            print(f"{key}: {value}")

        print("\n========== MISSING VALUES ==========\n")

        print(loader.count_missing_values(df))

        print("\n========== COLUMN TYPES ==========\n")

        column_types = loader.detect_column_types(df)

        for column, column_type in column_types.items():
            print(f"{column}: {column_type}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
