from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler
from tools.column_classifier import ColumnClassifier


def test_dataset(file_path):
    print("\n" + "=" * 70)
    print(f"Testing Dataset : {file_path}")
    print("=" * 70)

    # -----------------------------
    # Load Dataset
    # -----------------------------
    loader = DataLoader()
    df = loader.load_file(file_path)

    # -----------------------------
    # Profile Dataset
    # -----------------------------
    profiler = DataProfiler()
    profile = profiler.run(df)

    # -----------------------------
    # Classify Columns
    # -----------------------------
    classifier = ColumnClassifier()
    classification = classifier.classify(df)

    # -----------------------------
    # Display Results
    # -----------------------------
    print("\n========== BASIC INFORMATION ==========\n")
    print(profile["basic_info"])

    print("\n========== MISSING VALUES ==========\n")
    print(profile["missing_values"])

    print("\n========== SUMMARY STATISTICS ==========\n")
    print(profile["summary_statistics"])

    print("\n========== DUPLICATE ROWS ==========\n")
    print(profile["duplicate_rows"])

    print("\n========== UNIQUE VALUES ==========\n")
    print(profile["unique_values"])

    print("\n========== COLUMN CLASSIFICATION ==========\n")

    for column, details in classification.items():
        print(f"{column}")

        for key, value in details.items():
            print(f"   {key}: {value}")

        print()


def main():
    # Test current dataset
    test_dataset("data/train.csv")


if __name__ == "__main__":
    main()