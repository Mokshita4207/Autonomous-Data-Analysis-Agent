from tools.code_executor import CodeExecutor
from config.llm import llm


class SelfCorrector:
    """
    Fixes generated Python code when execution fails.
    """

    def __init__(self, executor=None):
        self.llm = llm
        self.executor = executor or CodeExecutor()

    # Validate generated code and execution result
    def validate_input(self, generated_code, execution_result):

        if not isinstance(generated_code, dict):
            raise TypeError("Generated code must be a dictionary.")

        if not isinstance(execution_result, dict):
            raise TypeError("Execution result must be a dictionary.")

        if execution_result.get("error") is None:
            raise ValueError("No execution error found to correct.")

        return True

    # Fix generated code using Gemini
    def fix_code(self, generated_code, execution_result):

        self.validate_input(
            generated_code,
            execution_result
        )

        if self.llm is None:
            raise ValueError("LLM is not initialized.")

        prompt = f"""
You are an expert Python developer.

The following generated Python code failed during execution.

Original Code:
{generated_code["code"]}

Execution Error:
{execution_result["error"]}

Fix the code.

Requirements:
- Return ONLY valid Python code.
- Do not return Markdown.
- Do not explain anything.
- Keep dataframe variable name as df.
- Store the final output in variable result.
"""

        response = self.llm.invoke(prompt)

        # Extract Python code from Gemini response
        if isinstance(response.content, list):

            corrected_code = ""

            for item in response.content:

                if isinstance(item, dict):

                    if item.get("type") == "text":

                        corrected_code += item.get("text", "")

                else:

                    corrected_code += str(item)

        else:

            corrected_code = str(response.content)

        return {
            "analysis_id": generated_code["analysis_id"],
            "analysis_type": generated_code["analysis_type"],
            "code": corrected_code.strip()
        }

    # Retry execution after fixing generated code
    def retry_execution(self, generated_code, df, max_retries=3):

        current_code = generated_code

        for attempt in range(max_retries):

            print(f"\nRetry Attempt {attempt + 1}")

            execution_result = self.executor.run(
                current_code,
                df
            )

            if execution_result["status"] == "Success":

                print("Execution Successful.")

                return execution_result

            print("Execution Failed.")

            print(execution_result["error"])

            current_code = self.fix_code(
                current_code,
                execution_result
            )

        return {
            "status": "Failed",
            "result": None,
            "error": "Maximum retry attempts reached."
        }

    # Run complete workflow
    def run(self, generated_code, df):

        return self.retry_execution(
            generated_code,
            df
        )


if __name__ == "__main__":

    import pandas as pd

    df = pd.DataFrame({
        "age": [20, 30, 40, 50]
    })

    generated_code = {
        "analysis_id": "A001",
        "analysis_type": "univariate_numeric",
        "code": """
result = {
    "mean": float(df["ages"].mean())
}
"""
    }

    corrector = SelfCorrector()

    result = corrector.run(
        generated_code,
        df
    )

    print("\nFinal Result")
    print(result)