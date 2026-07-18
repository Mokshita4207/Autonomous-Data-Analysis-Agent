import json
import textwrap
import traceback

from e2b_code_interpreter import Sandbox
from config.e2b_config import E2B_API_KEY

class CodeExecutor:
    """
    Executes generated Python code inside an E2B Sandbox.
    """

    def __init__(self):
        pass

    # ---------------------------------
    # Validate Input
    # ---------------------------------
    def validate_input(self, generated_code):

        if not isinstance(generated_code, dict):
            raise TypeError("Input must be a dictionary.")

        if "analysis_id" not in generated_code:
            raise ValueError("Missing analysis_id.")

        if "analysis_type" not in generated_code:
            raise ValueError("Missing analysis_type.")

        if "code" not in generated_code:
            raise ValueError("Missing generated code.")

        if generated_code["code"] is None:
            raise ValueError("Generated code is empty.")

        return True

    # ---------------------------------
    # Execute Code in E2B Sandbox
    # ---------------------------------
    def execute_code(self, generated_code, df):

    # Validate input
        self.validate_input(generated_code)

        try:

            # Remove indentation from generated code
            generated_python = textwrap.dedent(
                generated_code["code"]
            )

            # Create Sandbox
            with Sandbox.create(api_key=E2B_API_KEY) as sandbox:

                # Upload dataframe
                sandbox.files.write(
                    "/home/user/data.csv",
                    df.to_csv(index=False)
                )

                # Complete Python script
                code = (
                    "import json\n"
                    "import pandas as pd\n\n"
                    "df = pd.read_csv('/home/user/data.csv')\n\n"
                    f"{generated_python}\n\n"
                    "print(json.dumps(result))"
                    )
                # Execute code
                execution = sandbox.run_code(code)

                # Check execution error
                if execution.error is not None:
                    raise Exception(execution.error)

                # Read printed output
                output = "".join(execution.logs.stdout).strip()

                # Convert JSON string to dictionary
                try:
                    result = json.loads(output)

                except Exception:
                    result = output

                return {
                    "success": True,
                    "result": result,
                    "error": None
                }

        except Exception:

            return {
                "success": False,
                "result": None,
                "error": traceback.format_exc()
            }
    # ---------------------------------
    # Format Result
    # ---------------------------------
    def get_execution_result(self, execution_result):

        if execution_result["success"]:

            return {
                "status": "Success",
                "result": execution_result["result"],
                "error": None
            }

        return {
            "status": "Failed",
            "result": None,
            "error": execution_result["error"]
        }

    # ---------------------------------
    # Complete Workflow
    # ---------------------------------
    def run(self, generated_code, df):

        execution_result = self.execute_code(
            generated_code,
            df
        )

        return self.get_execution_result(
            execution_result
        )