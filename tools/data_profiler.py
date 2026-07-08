import pandas as pd

class DataProfiler:
    """
    Generates summary information about a dataset.
    """
    def validate_dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Dataset is empty.")
        return True
    
    def get_basic_info(self, df):
        self.validate_dataframe(df)

        info = {
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Shape": df.shape,
            "Column Names": list(df.columns),
            "Data Types": df.dtypes.astype(str).to_dict(),
            "Memory Usage (KB)": float(round(df.memory_usage(deep=True).sum()/1024,2)),
        }
        return info
    
    # Generate summary statistics using df.describe(include="all")
    # Returns count, mean, std, min, max, quartiles, and categorical summaries.
    def summary_statistics(self, df):
        self.validate_dataframe(df)

        statistics = df.describe(include="all")

        return statistics
    
    # Count missing values in each column
    def count_missing_values(self, df):
        self.validate_dataframe(df)

        missing = pd.DataFrame({
            "Missing Values": df.isnull().sum(),
            "Missing Percentage": round(df.isnull().mean() * 100, 2)
        })

        return missing
    
    # Find unique values and their frequency
    def unique_value_distribution(self, df):
        self.validate_dataframe(df)

        unique_distribution = {}
        for column in df.columns:
            distribution = df[column].value_counts(dropna=False)
            unique_distribution[column] = distribution.to_dict()
                
        return unique_distribution
    
     # Find duplicate rows
    def duplicate_rows(self, df):
        self.validate_dataframe(df)

        duplicate_count = df.duplicated().sum()

        duplicate_data = df[df.duplicated()]

        return {
            "Duplicate Count": duplicate_count,
            "Duplicate Rows": duplicate_data
        }

    # Run all functions together
    def run(self, df):

        return {
            "basic_info": self.get_basic_info(df),
            "missing_values": self.count_missing_values(df),
            "summary_statistics": self.summary_statistics(df),
            "duplicate_rows": self.duplicate_rows(df),
            "unique_values": self.unique_value_distribution(df)
        }

   