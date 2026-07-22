import pytest

from agents.iterative_analysis_agent import (
    IterativeAnalysisAgent
)


def test_validate_input():

    agent = IterativeAnalysisAgent()

    assert (
        agent.validate_input(
            [
                {
                    "analysis_id": "A001"
                }
            ]
        )
        is True
    )


def test_invalid_input():

    agent = IterativeAnalysisAgent()

    with pytest.raises(
        TypeError
    ):

        agent.validate_input(
            "invalid"
        )


def test_build_prompt():

    agent = IterativeAnalysisAgent()

    prompt = agent.build_prompt(

        [

            {
                "analysis_type":
                    "correlation_matrix"
            }

        ],

        [

            "Strong relationship detected."

        ]

    )

    assert isinstance(
        prompt,
        str
    )

    assert (
        "follow-up"
        in prompt
    )


def test_no_llm():

    agent = IterativeAnalysisAgent()

    agent.llm = None

    result = (
        agent.generate_follow_up_tasks(
            [],
            []
        )
    )

    assert result == []