import pandas as pd

from agents.self_corrector import SelfCorrector
from tools.code_executor import CodeExecutor

# ---------------------------------------------------------
# Sample dataframe
# ---------------------------------------------------------
df = pd.DataFrame({
    "age": [20, 30, 40, 50]
})

# ---------------------------------------------------------
# Create CodeExecutor
# ---------------------------------------------------------
executor = CodeExecutor()

# ---------------------------------------------------------
# NOTE:
# Replace 'None' with your Gemini LLM object when available.
# Example:
#
# from langchain_google_genai import ChatGoogleGenerativeAI
#
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key="YOUR_API_KEY"
# )
#
# corrector = SelfCorrector(llm)
# ---------------------------------------------------------

corrector = SelfCorrector(None)

# =========================================================
# TEST 1 : VALID CODE
# =========================================================

print("=" * 70)
print("TEST 1 : VALID GENERATED CODE")
print("=" * 70)

generated_code = {
    "analysis_id": "A001",
    "analysis_type": "univariate_numeric",
    "code": """
result = {
    "mean": float(df["age"].mean())
}
"""
}

result = executor.run(
    generated_code,
    df
)

print(result)

# =========================================================
# TEST 2 : INVALID COLUMN
# =========================================================

print("\n" + "=" * 70)
print("TEST 2 : INVALID COLUMN")
print("=" * 70)

generated_code = {
    "analysis_id": "A002",
    "analysis_type": "univariate_numeric",
    "code": """
result = {
    "mean": float(df["ages"].mean())
}
"""
}

result = executor.run(
    generated_code,
    df
)

print(result)

# =========================================================
# TEST 3 : SELF CORRECTOR
# =========================================================

print("\n" + "=" * 70)
print("TEST 3 : SELF CORRECTOR")
print("=" * 70)

try:

    corrected = corrector.run(
        generated_code,
        df
    )

    print(corrected)

except Exception as e:

    print(e)

# =========================================================
# TEST 4 : MISSING CODE
# =========================================================

print("\n" + "=" * 70)
print("TEST 4 : MISSING CODE")
print("=" * 70)

generated_code = {
    "analysis_id": "A003",
    "analysis_type": "univariate_numeric"
}

try:

    print(
        executor.run(
            generated_code,
            df
        )
    )

except Exception as e:

    print(e)

# =========================================================
# TEST 5 : EMPTY CODE
# =========================================================

print("\n" + "=" * 70)
print("TEST 5 : EMPTY CODE")
print("=" * 70)

generated_code = {
    "analysis_id": "A004",
    "analysis_type": "univariate_numeric",
    "code": None
}

try:

    print(
        executor.run(
            generated_code,
            df
        )
    )

except Exception as e:

    print(e)

# =========================================================
# TEST 6 : INVALID INPUT TYPE
# =========================================================

print("\n" + "=" * 70)
print("TEST 6 : INVALID INPUT")
print("=" * 70)

try:

    print(
        executor.run(
            "Hello",
            df
        )
    )

except Exception as e:

    print(e)