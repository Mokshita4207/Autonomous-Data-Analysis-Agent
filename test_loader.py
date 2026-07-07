from tools.data_loader import DataLoader

loader = DataLoader()

try:

    df = loader.load_file("data/sample.csv")

    print(loader.validate_file(df))

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
    print(e)