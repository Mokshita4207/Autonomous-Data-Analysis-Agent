"""
IterativeAnalysisAgent

Performs deeper analysis by generating follow-up questions
from existing analysis results and insights.
"""

import json
import logging

from config.llm import llm

logger = logging.getLogger(__name__)


class IterativeAnalysisAgent:

    def __init__(
        self,
        max_iterations=2
    ):

        self.llm = llm
        self.max_iterations = max_iterations

    def validate_input(
        self,
        analysis_results
    ):

        if not isinstance(
            analysis_results,
            list
        ):

            raise TypeError(
                "Analysis results must be a list."
            )

        if not analysis_results:

            raise ValueError(
                "Analysis results cannot be empty."
            )

        return True

    def build_prompt(
        self,
        analysis_results,
        insights
    ):

        prompt = f"""
You are an expert data scientist.

The dataset has already been analyzed.

Previous analysis results:

{analysis_results}

Previous insights:

{insights}

Your task is to identify useful follow-up analyses.

Only suggest follow-up analysis when the previous results
show a meaningful relationship, difference, trend, anomaly,
or unexpected pattern.

Do not repeat an analysis that has already been performed.

Return a JSON list.

Each item must contain:

{{
    "analysis_type": "...",
    "columns": ["column1", "column2"],
    "rationale": "Why this deeper analysis is useful"
}}

Allowed analysis types:

- univariate_numeric
- univariate_categorical
- correlation_matrix
- bivariate_numeric_numeric
- segment_comparison
- trend_over_time
- anomaly_detection
- target_relationship

Return at most 2 follow-up analyses.

Return only valid JSON.
"""

        return prompt

    def generate_follow_up_tasks(
        self,
        analysis_results,
        insights
    ):

        if self.llm is None:

            logger.warning(
                "LLM unavailable for iterative analysis."
            )

            return []

        prompt = self.build_prompt(
            analysis_results,
            insights
        )

        response = self.llm.invoke(
            prompt
        )

        content = getattr(
            response,
            "content",
            None
        )

        if content is None:

            return []

        content = str(
            content
        ).strip()

        if content.startswith(
            "```"
        ):

            content = (
                content
                .replace(
                    "```json",
                    ""
                )
                .replace(
                    "```",
                    ""
                )
                .strip()
            )

        try:

            tasks = json.loads(
                content
            )

        except json.JSONDecodeError:

            logger.error(
                "LLM returned invalid JSON."
            )

            return []

        if not isinstance(
            tasks,
            list
        ):

            return []

        valid_tasks = []

        for task in tasks:

            if not isinstance(
                task,
                dict
            ):

                continue

            if (
                "analysis_type"
                not in task
            ):

                continue

            if (
                "columns"
                not in task
            ):

                continue

            if (
                "rationale"
                not in task
            ):

                continue

            valid_tasks.append(
                task
            )

        return valid_tasks[:2]

    def run(
        self,
        analysis_results,
        insights
    ):

        self.validate_input(
            analysis_results
        )

        iterative_results = []

        current_results = (
            analysis_results
        )

        current_insights = (
            insights
        )

        for iteration in range(
            1,
            self.max_iterations + 1
        ):

            logger.info(
                "Starting iteration %s",
                iteration
            )

            follow_up_tasks = (
                self.generate_follow_up_tasks(
                    current_results,
                    current_insights
                )
            )

            if not follow_up_tasks:

                logger.info(
                    "No further analysis required."
                )

                break

            iteration_result = {

                "iteration":
                    iteration,

                "follow_up_tasks":
                    follow_up_tasks

            }

            iterative_results.append(
                iteration_result
            )

            current_results = (
                follow_up_tasks
            )

            current_insights = []

        return {

            "iterations":
                iterative_results,

            "total_iterations":
                len(
                    iterative_results
                )

        }


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    sample_results = [

        {

            "analysis_id":
                "A001",

            "analysis_type":
                "correlation_matrix",

            "result": {

                "age": {

                    "cholesterol":
                        0.75

                }

            }

        }

    ]

    sample_insights = [

        "Age and cholesterol show a strong positive relationship."

    ]

    agent = IterativeAnalysisAgent()

    result = agent.run(

        sample_results,

        sample_insights

    )

    print(result)