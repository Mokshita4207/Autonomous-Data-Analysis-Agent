"""
CodeGeneratorAgent

Converts AnalysisPlanner output into executable Python analysis code.
Uses predefined templates for reliability and LLM fallback for unknown tasks.
"""

import logging

logger = logging.getLogger(__name__)


class CodeGeneratorAgent:

    def __init__(self, llm=None):
        self.llm = llm


    def generate_code(self, analysis_task):

        analysis_type = analysis_task["analysis_type"]

        try:

            if analysis_type == "data_quality":
                code = self.data_quality_code(
                    analysis_task
                )

            elif analysis_type == "univariate_numeric":
                code = self.univariate_numeric_code(
                    analysis_task
                )

            elif analysis_type == "univariate_categorical":
                code = self.univariate_categorical_code(
                    analysis_task
                )

            elif analysis_type == "correlation_matrix":
                code = self.correlation_code(
                    analysis_task
                )

            elif analysis_type == "bivariate_numeric_numeric":
                code = self.bivariate_code(
                    analysis_task
                )

            elif analysis_type == "segment_comparison":
                code = self.segment_code(
                    analysis_task
                )

            elif analysis_type == "trend_over_time":
                code = self.trend_code(
                    analysis_task
                )

            elif analysis_type == "anomaly_detection":
                code = self.anomaly_code(
                    analysis_task
                )

            else:
                code = self.llm_generate(
                    analysis_task
                )


            return {
                "analysis_id": analysis_task["analysis_id"],
                "analysis_type": analysis_type,
                "code": code
            }


        except Exception as e:

            logger.error(
                "Code generation failed: %s",
                e
            )

            return {
                "analysis_id": analysis_task["analysis_id"],
                "analysis_type": analysis_type,
                "code": None,
                "error": str(e)
            }



    def data_quality_code(self, task):

        return """
result = {
    "missing_values": df.isnull().sum().to_dict(),
    "duplicate_rows": int(df.duplicated().sum())
}
"""


    def univariate_numeric_code(self, task):

        col = task["columns"][0]

        return f"""
result = {{
    "column": "{col}",
    "mean": float(df["{col}"].mean()),
    "median": float(df["{col}"].median()),
    "std": float(df["{col}"].std()),
    "min": float(df["{col}"].min()),
    "max": float(df["{col}"].max())
}}
"""


    def univariate_categorical_code(self, task):

        col = task["columns"][0]

        return f"""
result = {{
    "column": "{col}",
    "value_counts": df["{col}"].value_counts().to_dict()
}}
"""


    def correlation_code(self, task):

        cols = task["columns"]

        return f"""
columns = {cols}

result = {{
    "correlation_matrix":
    df[columns].corr().to_dict()
}}
"""


    def bivariate_code(self, task):

        c1 = task["columns"][0]
        c2 = task["columns"][1]

        return f"""
result = {{
    "columns": ["{c1}", "{c2}"],
    "correlation":
    float(df["{c1}"].corr(df["{c2}"]))
}}
"""


    def segment_code(self, task):

        group = task["columns"][0]
        value = task["columns"][1]

        return f"""
result = (
    df.groupby("{group}")["{value}"]
    .mean()
    .to_dict()
)
"""


    def trend_code(self, task):

        date = task["columns"][0]
        value = task["columns"][1]

        return f"""
result = (
    df.groupby("{date}")["{value}"]
    .mean()
    .to_dict()
)
"""


    def anomaly_code(self, task):

        col = task["columns"][0]

        return f"""
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

prediction = model.fit_predict(
    df[["{col}"]]
)

result = {{
    "column": "{col}",
    "outlier_count":
    int((prediction == -1).sum())
}}
"""


    def llm_generate(self, task):

        if self.llm is None:
            raise Exception(
                "No LLM available for unknown analysis"
            )


        prompt = f"""
Generate pandas/scipy Python analysis code.

Analysis:
{task}

Requirements:
- dataframe variable name is df
- store final output in variable result
- only return Python code
"""

        response = self.llm.invoke(prompt)

        return response.content
    
if __name__ == "__main__":

    generator = CodeGeneratorAgent()

    task = {
        "analysis_id": "A001",
        "analysis_type": "univariate_numeric",
        "columns": ["age"],
        "rationale": "Summarize age distribution",
        "priority": 5
    }

    result = generator.generate_code(task)

    print(result) 