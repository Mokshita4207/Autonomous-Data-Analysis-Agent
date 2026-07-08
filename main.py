import pandas as pd
from tools.data_profiler import DataProfiler

# Temporary: Read dataset directly
df = pd.read_csv("data/sample.csv")

profiler = DataProfiler()

result = profiler.run(df)

print(result)