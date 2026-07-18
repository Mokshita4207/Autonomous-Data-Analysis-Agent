import pandas as pd

from tools.code_executor import CodeExecutor


df = pd.DataFrame({
    "age": [20, 30, 40, 50]
})

generated_code = {
    "analysis_id": "A001",
    "analysis_type": "univariate_numeric",
    "code": """
result = {
    "mean": float(df["age"].mean()),
    "max": float(df["age"].max()),
    "min": float(df["age"].min())
}
"""
}

executor = CodeExecutor()

result = executor.run(
    generated_code,
    df
)

print(result)