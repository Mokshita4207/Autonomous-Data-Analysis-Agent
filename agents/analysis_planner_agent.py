"""
AnalysisPlannerAgent
Consumes ColumnClassifier + DataProfiler output only. Recomputes nothing.
"""

import itertools
import logging
from typing import List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PRIORITY = {
    "data_quality": 0,
    "target_relationship": 1,
    "trend_over_time": 2,
    "correlation_matrix": 2,
    "segment_comparison": 3,
    "bivariate_numeric_numeric": 4,
    "univariate_numeric": 5,
    "univariate_categorical": 5,
    "anomaly_detection": 6,
    "general_profiling": 5,
}

MAX_ANALYSES = 10
MAX_PAIRWISE_ANALYSES = 3
MAX_SEGMENT_GROUP_COLS = 2
MAX_SEGMENT_NUMERIC_COLS = 2
MAX_TARGET_RELATIONSHIPS = 5

ALLOWED_GOALS = [
    "data_quality", "explain_target", "trend_over_time",
    "segment_comparison", "find_relationships",
    "investigate_anomalies", "general_profiling",
]

DEFAULT_TASK = {
    "analysis_id": "A001",
    "analysis_type": "general_profiling",
    "columns": [],
    "rationale": "Basic dataset overview",
    "priority": PRIORITY["general_profiling"],
}


class GoalResponse(BaseModel):
    goals: List[str] = Field(
        description=f"Subset of these allowed goals only: {ALLOWED_GOALS}"
    )
    reasoning: str = Field(description="One sentence explaining the choice")


class AnalysisTask(BaseModel):
    """Fixed output schema every downstream tool (CodeGenerator,
    ChartGenerator, ReportBuilder) can rely on."""
    analysis_id: str
    analysis_type: str
    columns: List[str]
    rationale: str
    priority: int


class AnalysisPlannerAgent:

    def __init__(self, llm=None):
        self.llm = llm or self._default_llm()

    def _default_llm(self):
        try:
            from dotenv import load_dotenv
            from langchain_google_genai import ChatGoogleGenerativeAI
            load_dotenv()
            return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        except Exception:
            return None

    def run(self, column_types, profile):
        dataset = self.inspect_dataset(column_types, profile)
        goals = self.identify_goals(dataset)
        candidates = self.select_analyses(dataset, goals)
        prioritized = self.prioritize_tasks(candidates)
        return self.create_plan(prioritized)

    def inspect_dataset(self, column_types, profile):
        """Deterministic summary from classifier + profiler output. No LLM here."""
        buckets = {"numeric": [], "categorical": [], "date": [], "target": [], "grouping": []}

        for col, info in column_types.items():
            t = info["type"] if isinstance(info, dict) else info
            role = info.get("role_hint") if isinstance(info, dict) else None

            # Case-insensitive matching — ColumnClassifier output casing
            # shouldn't be able to break the planner.
            t_normalized = t.lower() if isinstance(t, str) else t
            role_normalized = role.lower() if isinstance(role, str) else role

            if t_normalized == "numeric":
                buckets["numeric"].append(col)
            elif t_normalized == "categorical":
                buckets["categorical"].append(col)
            elif t_normalized == "date":
                buckets["date"].append(col)

            if role_normalized == "target":
                buckets["target"].append(col)
            elif role_normalized == "grouping":
                buckets["grouping"].append(col)

        stats = (profile or {}).get("summary_statistics", {})
        buckets["outliers"] = [c for c, s in stats.items()
                                if isinstance(s, dict) and s.get("outlier_count", 0) > 0]

        missing = (profile or {}).get("missing_values", {})
        missing_counts = missing.get("Missing Values", {}) if isinstance(missing, dict) else {}
        buckets["has_missing"] = any(v > 0 for v in missing_counts.values())

        dup = (profile or {}).get("duplicates", {})
        buckets["has_duplicates"] = dup.get("duplicate_row_count", 0) > 0

        buckets["n_numeric"] = len(buckets["numeric"])
        buckets["n_categorical"] = len(buckets["categorical"])
        buckets["n_targets"] = len(buckets["target"])
        buckets["has_dates"] = len(buckets["date"]) > 0

        return buckets

    def identify_goals(self, dataset):
        if self.llm is None:
            return self._identify_goals_rule_based(dataset)

        try:
            structured_llm = self.llm.with_structured_output(GoalResponse)
            prompt = f"""
You are an AI Analysis Planning Agent.

Your task is to analyze the dataset summary and determine which analysis goals are appropriate.

Dataset Summary:
{dataset}

Allowed Goals:
{ALLOWED_GOALS}

Rules:
- Only choose goals from the allowed list.
- Do not invent new goals.
- Select every goal that genuinely applies.
- Return the reasoning for your decision.
"""
            result: GoalResponse = structured_llm.invoke(prompt)
            logger.info("LLM goals: %s | reason: %s", result.goals, result.reasoning)

            goals = [g for g in result.goals if g in ALLOWED_GOALS]
            if len(goals) == 0:
                return ["general_profiling"]
            return goals
        except Exception as e:
            logger.error("LLM error during goal identification: %s", e)
            return self._identify_goals_rule_based(dataset)

    def _identify_goals_rule_based(self, dataset):
        d = dataset
        goals = []
        if d["has_missing"] or d["has_duplicates"]:
            goals.append("data_quality")
        if d["n_targets"] > 0:
            goals.append("explain_target")
        if d["has_dates"] and d["n_numeric"] > 0:
            goals.append("trend_over_time")
        if d["grouping"] or (d["n_categorical"] > 0 and d["n_numeric"] > 0):
            goals.append("segment_comparison")
        if d["n_numeric"] >= 2:
            goals.append("find_relationships")
        if d["outliers"]:
            goals.append("investigate_anomalies")
        if not goals:
            goals.append("general_profiling")
        return goals

    def select_analyses(self, dataset, goals):
        """Generate candidate analyses for each identified goal. Deterministic."""
        d = dataset
        candidates = []

        # Numeric/categorical columns excluding targets — prevents the
        # same column pair being analyzed once as target_relationship
        # and again as a generic bivariate/correlation candidate.
        numeric_non_target = [c for c in d["numeric"] if c not in d["target"]]
        categorical_non_target = [c for c in d["categorical"] if c not in d["target"]]

        if "data_quality" in goals:
            if d["has_missing"]:
                candidates.append({"analysis_type": "data_quality", "columns": [],
                                    "rationale": "Assess missing value patterns across the dataset"})
            if d["has_duplicates"]:
                candidates.append({"analysis_type": "data_quality", "columns": [],
                                    "rationale": "Check for and quantify duplicate rows"})

        if "explain_target" in goals:
            count = 0
            for target in d["target"]:
                for col in numeric_non_target + categorical_non_target:
                    if count >= MAX_TARGET_RELATIONSHIPS:
                        break
                    candidates.append({"analysis_type": "target_relationship",
                                        "columns": [col, target],
                                        "rationale": f"How does {col} relate to {target}?"})
                    count += 1

        if "trend_over_time" in goals and d["date"]:
            date_col = d["date"][0]
            for num in numeric_non_target:
                candidates.append({"analysis_type": "trend_over_time",
                                    "columns": [date_col, num],
                                    "rationale": f"Trend of {num} over {date_col}"})

        if "segment_comparison" in goals:
            group_cols = (d["grouping"] or categorical_non_target)[:MAX_SEGMENT_GROUP_COLS]
            for g in group_cols:
                for num in numeric_non_target[:MAX_SEGMENT_NUMERIC_COLS]:
                    candidates.append({"analysis_type": "segment_comparison",
                                        "columns": [g, num],
                                        "rationale": f"Compare {num} across {g} segments"})

        if "find_relationships" in goals and len(numeric_non_target) >= 2:
            candidates.append({"analysis_type": "correlation_matrix", "columns": numeric_non_target,
                                "rationale": "Correlation across numeric columns"})
            pairs = list(itertools.combinations(numeric_non_target, 2))[:MAX_PAIRWISE_ANALYSES]
            for c1, c2 in pairs:
                candidates.append({"analysis_type": "bivariate_numeric_numeric",
                                    "columns": [c1, c2],
                                    "rationale": f"Relationship between {c1} and {c2}"})

        if "investigate_anomalies" in goals:
            for col in d["outliers"]:
                candidates.append({"analysis_type": "anomaly_detection", "columns": [col],
                                    "rationale": f"Investigate outliers in {col}"})

        if "general_profiling" in goals:
            for col in numeric_non_target:
                candidates.append({"analysis_type": "univariate_numeric", "columns": [col],
                                    "rationale": f"Summarize distribution of {col}"})
            for col in categorical_non_target:
                candidates.append({"analysis_type": "univariate_categorical", "columns": [col],
                                    "rationale": f"Summarize categories of {col}"})

        return self._deduplicate(candidates)

    def _deduplicate(self, candidates):
        """Removes exact duplicates: same analysis_type + same column set."""
        unique = []
        seen = set()
        for c in candidates:
            key = (c["analysis_type"], tuple(c["columns"]))
            if key not in seen:
                unique.append(c)
                seen.add(key)
        return unique

    def prioritize_tasks(self, candidates):
        for c in candidates:
            c["priority"] = PRIORITY.get(c["analysis_type"], 9)
        ordered = sorted(candidates, key=lambda c: c["priority"])[:MAX_ANALYSES]

        if not ordered:
            return [dict(DEFAULT_TASK)]  # copy, so callers can't mutate the shared default
        return ordered

    def create_plan(self, prioritized_candidates):
        """Assigns final analysis_id and validates each task against the
        fixed AnalysisTask schema before returning."""
        plan = []
        for i, c in enumerate(prioritized_candidates, start=1):
            task = AnalysisTask(
                analysis_id=c.get("analysis_id", f"A{i:03d}"),
                analysis_type=c["analysis_type"],
                columns=c["columns"],
                rationale=c["rationale"],
                priority=c["priority"],
            )
            plan.append(task.model_dump())
        return plan