import logging

from config.llm import llm

logger = logging.getLogger(__name__)


class InsightExtractor:
    """
    Generates natural language insights from execution results.
    """

    def __init__(self):
        self.llm = llm

    # Validate input
    def validate_input(self, execution_result):
        """Validate that execution_result has the required structure."""

        if not isinstance(execution_result, dict):
            raise TypeError("Execution result must be a dictionary.")

        if "analysis_id" not in execution_result:
            raise ValueError("Execution result is missing analysis_id.")

        if "analysis_type" not in execution_result:
            raise ValueError("Execution result is missing analysis_type.")

        if "status" not in execution_result:
            raise ValueError("Execution result is missing status.")

        if "result" not in execution_result:
            raise ValueError("Execution result is missing result.")

        if execution_result.get("status") != "Success":
            raise ValueError("Execution result status must be Success.")

        if not isinstance(execution_result.get("result"), dict):
            raise TypeError("Execution result must contain a result dictionary.")

        return True

    # Build prompt
    def build_prompt(self, analysis_type, result):
        """Build a senior-analyst style Gemini prompt for the analysis type."""

        if analysis_type == "univariate_numeric":
            instruction = "Interpret what the mean, median and spread imply about the population, and mention any skew, concentration or risk."
        elif analysis_type == "univariate_categorical":
            instruction = "Interpret what the dominant categories imply about concentration, diversity or imbalance in the data."
        elif analysis_type == "bivariate_numeric_numeric":
            instruction = "Interpret what the relationship between the two columns implies, including possible cause, dependency or business impact."
        elif analysis_type == "correlation_matrix":
            instruction = "Interpret what the strongest relationships imply about how the variables move together and why that matters."
        elif analysis_type == "segment_comparison":
            instruction = "Interpret why one segment leads and one lags, and what opportunity or risk that gap implies for the business."
        elif analysis_type == "trend_over_time":
            instruction = "Interpret what the trend direction implies about future performance, momentum or seasonality."
        elif analysis_type == "anomaly_detection":
            instruction = "Interpret what the number of anomalies implies about data quality or process stability, not just the count."
        elif analysis_type == "target_relationship":
            instruction = "Interpret why the strongest feature matters for the target and what that implies for decision making."
        else:
            instruction = "Interpret the key findings and what they imply."

        prompt = f"""
You are an experienced Data Scientist.

Your task is to interpret statistical analysis results.

Base every conclusion strictly on the provided results.

Never invent context or assume the dataset domain.

Analysis type: {analysis_type}
Result data: {result}

Treat the result dictionary as the ONLY source of truth.

If information is missing, explicitly state that it cannot be inferred.

Do not assume the dataset contains customers, products, sales, finance, healthcare, or any other domain unless it is explicitly present.

{instruction}

Requirements:
- Generate exactly 2 to 3 insights.
- Base every insight ONLY on the provided analysis results.
- Do NOT assume the dataset domain (customers, healthcare, finance, sales, etc.) unless explicitly mentioned.
- Do NOT invent facts that are not present in the result.
- Explain what the statistics imply about the data distribution, relationships, trends or variability.
- Mention skewness, spread, concentration, correlation strength or anomalies when supported by the data.
- If there is not enough information, say so instead of making assumptions.
- Keep each insight under 30 words.
- Return each insight on its own line.
- Do not use Markdown, numbering or bullet points.
- Do not include any introduction or conclusion.
"""

        return prompt

    # Generate insights using Gemini
    def generate_insights(self, prompt):
        """Call Gemini and extract a list of insights from the response."""

        logger.info("Generating insights using Gemini.")

        response = self.llm.invoke(prompt)
        content = getattr(response, "content", None)

        if content is None:
            raise ValueError("Gemini response has no content.")

        if isinstance(content, list):
            text = ""
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text += item.get("text", "")
                elif item is not None:
                    text += str(item)
        else:
            text = str(content)

        insights = [
            line.strip(" -*\t")
            for line in text.strip().splitlines()
            if line.strip(" -*\t")
        ]

        if not insights:
            raise ValueError("Gemini returned no usable insights.")

        return insights[:3]

    # Interpret correlation strength into readable text
    def _describe_correlation(self, value):
        """Convert a numeric correlation value into readable text."""

        try:
            value = float(value)
        except (TypeError, ValueError):
            return "unknown correlation"

        if value >= 0.90:
            label = "Very strong positive correlation"
        elif value >= 0.70:
            label = "Strong positive correlation"
        elif value >= 0.40:
            label = "Moderate positive correlation"
        elif value >= 0.10:
            label = "Weak positive correlation"
        elif value >= -0.10:
            label = "No significant correlation"
        elif value >= -0.39:
            label = "Weak negative correlation"
        elif value >= -0.69:
            label = "Moderate negative correlation"
        elif value >= -0.89:
            label = "Strong negative correlation"
        else:
            label = "Very strong negative correlation"

        return label

    # Fallback if Gemini fails
    def fallback_insights(self, analysis_type, result):
        """Generate human-readable, rule-based insights when Gemini fails."""

        insights = []

        if analysis_type == "univariate_numeric":
            column = result.get("column", "the variable")

            if "mean" in result:
                insights.append(f"The average {column} suggests a typical central value across the dataset.")

            if "mean" in result and "median" in result and result["mean"] != result["median"]:
                insights.append(f"A gap between mean and median {column} points to a skewed distribution.")

            if "min" in result and "max" in result:
                insights.append(f"The wide range in {column} indicates considerable variation across records.")
        elif analysis_type == "univariate_categorical":

            column = result.get("column", "the category")
            value_counts = result.get("value_counts", {})

            if value_counts:
                top_category = max(value_counts, key=value_counts.get)

                insights.append(
                    f"{top_category} is the most frequent category in {column}."
                )

                insights.append(
                    f"{len(value_counts)} unique categories are present in {column}."
                )
        elif analysis_type == "bivariate_numeric_numeric":

            columns = result.get("columns", [])

            col_x = columns[0] if len(columns) > 0 else "the first variable"
            col_y = columns[1] if len(columns) > 1 else "the second variable"

            if "correlation" in result:
                label = self._describe_correlation(result["correlation"])

                insights.append(
                    f"{label} suggests {col_x} and {col_y} tend to move together."
                )
        elif analysis_type == "correlation_matrix":

            corr = result.get("correlation_matrix", {})

            if corr:
                insights.append(
                    "The correlation matrix was generated successfully and summarizes relationships among numeric variables."
                )

                insights.append(
                    "Positive values indicate variables moving together, while negative values indicate inverse relationships."
                )
        elif analysis_type == "segment_comparison":
            if "top_segment" in result:
                insights.append(f"{result['top_segment']} is the dominant segment, driving the largest share of performance.")

            if "bottom_segment" in result:
                insights.append(f"{result['bottom_segment']} lags behind, signalling a possible growth opportunity.")
        elif analysis_type == "trend_over_time":
            if "trend_direction" in result:
                direction = result["trend_direction"]
                insights.append(f"The {direction} trend suggests momentum that is likely to continue near term.")
        elif analysis_type == "anomaly_detection":
            if "anomaly_count" in result:
                count = result["anomaly_count"]
                if count == 0:
                    insights.append("No anomalies were found, indicating the dataset is stable and consistent.")
                elif count <= 5:
                    insights.append("Only a few unusual observations were detected, indicating general consistency.")
                else:
                    insights.append("A notable number of anomalies were detected, suggesting a data quality risk.")
        elif analysis_type == "target_relationship":

            feature = result.get("feature")
            target = result.get("target")
            group_mean = result.get("group_mean")

            if feature and target and group_mean:
                insights.append(
                    f"{feature} shows different average values across {target} groups."
                )

                insights.append(
                    f"The relationship suggests {feature} may help distinguish target categories."
                )
        if not insights:
            insights.append("No meaningful interpretation could be derived from this analysis.")

        return insights

    # Run complete workflow
    def run(self, execution_result):
        """Validate, generate insights, fall back if needed, never raise."""

        try:
            self.validate_input(execution_result)

        except (TypeError, ValueError) as e:
            logger.error("Insight generation failed: %s", e)

            return {
                "analysis_id": execution_result.get("analysis_id")
                if isinstance(execution_result, dict) else None,
                "analysis_type": execution_result.get("analysis_type")
                if isinstance(execution_result, dict) else None,
                "status": "Failed",
                "insights": [],
                "error": str(e)
            }

        analysis_id = execution_result["analysis_id"]
        analysis_type = execution_result["analysis_type"]
        result = execution_result["result"]

        try:
            prompt = self.build_prompt(analysis_type, result)
            insights = self.generate_insights(prompt)

        except Exception as e:
            logger.warning("Gemini unavailable. Using fallback insights.")
            logger.error("Insight generation failed: %s", e)
            insights = self.fallback_insights(analysis_type, result)

        return {
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "status": "Success",
            "insights": insights
        }


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    execution_result = {
        "analysis_id": "A001",
        "analysis_type": "univariate_numeric",
        "status": "Success",
        "result": {
            "column": "age",
            "mean": 54.3,
            "median": 55,
            "std": 8.2,
            "min": 29,
            "max": 77
        }
    }

    extractor = InsightExtractor()
    output = extractor.run(execution_result)

    print("\nFinal Result")
    print(output)