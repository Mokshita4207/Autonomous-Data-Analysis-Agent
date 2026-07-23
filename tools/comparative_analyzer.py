import logging
import math

import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MIN_GROUP_SIZE = 5          # skip groups with too few rows to be meaningful
MAX_GROUPS_FOR_COMPARISON = 6  # skip columns with too many categories (not a real "group compare")


class ComparisonInsight(BaseModel):
    summary: str = Field(description="1-2 sentence plain-language summary of what the comparison shows")
    business_impact: str = Field(description="1-2 sentences on why this result matters in practice")
    recommendation: str = Field(description="One concrete, actionable recommendation based on the result")


class ComparativeAnalyzer:

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

    def run(self, df, column_types):
        """
        column_types: dict from ColumnClassifier.classify()
        Returns a list of comparison result dicts, each self-describing
        (type, columns involved, per-group stats, effect size, significance,
        and an AI-generated business interpretation under "ai_insight").
        """
        logger.info("Running comparative analysis")
        results = []
        results.extend(self._run_group_comparisons(df, column_types))
        results.extend(self._run_time_comparisons(df, column_types))

        for i, result in enumerate(results):
            result["comparison_id"] = f"cmp_{i + 1}"
            result["ai_insight"] = self._generate_ai_insight(result)

        logger.info("Comparative analysis complete: %d comparisons run", len(results))
        return results

    # ---- detection ----

    def detect_group_comparisons(self, column_types):
        """Categorical columns worth comparing groups on, plus numeric columns to compare."""
        categorical_cols, numeric_cols = [], []
        for col, info in column_types.items():
            t = (info.get("type") if isinstance(info, dict) else info)
            t = t.lower() if isinstance(t, str) else t
            if t == "categorical":
                categorical_cols.append(col)
            elif t == "numeric":
                numeric_cols.append(col)
        return categorical_cols, numeric_cols

    def detect_time_comparisons(self, column_types):
        date_cols = []
        for col, info in column_types.items():
            t = (info.get("type") if isinstance(info, dict) else info)
            t = t.lower() if isinstance(t, str) else t
            if t == "date":
                date_cols.append(col)
        return date_cols

    # ---- effect size ----

    def _cohens_d(self, a, b):
        """Cohen's d for two independent samples, using pooled standard deviation."""
        n1, n2 = len(a), len(b)
        var1, var2 = a.var(ddof=1), b.var(ddof=1)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((a.mean() - b.mean()) / pooled_std)

    def _eta_squared(self, value_lists):
        """Eta squared (proportion of variance explained) for 3+ group ANOVA."""
        all_values = pd.concat(value_lists)
        grand_mean = all_values.mean()
        ss_between = sum(len(v) * (v.mean() - grand_mean) ** 2 for v in value_lists)
        ss_total = float(((all_values - grand_mean) ** 2).sum())
        if ss_total == 0:
            return 0.0
        return float(ss_between / ss_total)

    def _effect_size(self, value_lists):
        """
        Returns (effect_size_type, effect_size, mean_difference).
        mean_difference is only meaningful for the 2-group case; for 3+
        groups we report the spread between the highest and lowest group
        means instead (mean_range), since there's no single "difference".
        """
        if len(value_lists) == 2:
            d = self._cohens_d(value_lists[0], value_lists[1])
            mean_diff = float(value_lists[0].mean() - value_lists[1].mean())
            return "cohens_d", round(d, 4), round(mean_diff, 4)
        else:
            eta_sq = self._eta_squared(value_lists)
            means = [v.mean() for v in value_lists]
            mean_range = float(max(means) - min(means))
            return "eta_squared", round(eta_sq, 4), round(mean_range, 4)

    # ---- group comparisons ----

    def _run_group_comparisons(self, df, column_types):
        categorical_cols, numeric_cols = self.detect_group_comparisons(column_types)
        results = []

        for group_col in categorical_cols:
            n_groups = df[group_col].nunique(dropna=True)
            if n_groups < 2 or n_groups > MAX_GROUPS_FOR_COMPARISON:
                continue

            for value_col in numeric_cols:
                result = self.compare_groups(df, group_col, value_col)
                if result:
                    results.append(result)

        return results

    def compare_groups(self, df, group_col, value_col):
        groups = {
            name: g[value_col].dropna()
            for name, g in df.groupby(group_col)
            if len(g[value_col].dropna()) >= MIN_GROUP_SIZE
        }
        if len(groups) < 2:
            return None

        group_stats = {
            str(name): {
                "count": int(len(vals)),
                "mean": round(float(vals.mean()), 4),
                "median": round(float(vals.median()), 4),
                "std": round(float(vals.std()), 4) if len(vals) > 1 else 0.0,
            }
            for name, vals in groups.items()
        }

        value_lists = list(groups.values())
        if len(value_lists) == 2:
            stat, p_value = stats.ttest_ind(value_lists[0], value_lists[1], equal_var=False)
            test_name = "welch_t_test"
        else:
            stat, p_value = stats.f_oneway(*value_lists)
            test_name = "anova"

        effect_size_type, effect_size, mean_diff = self._effect_size(value_lists)

        return {
            "comparison_type": "group",
            "group_column": group_col,
            "value_column": value_col,
            "group_stats": group_stats,
            "test": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "is_significant": bool(p_value < 0.05),
            "effect_size_type": effect_size_type,
            "effect_size": effect_size,
            "mean_difference": mean_diff,
        }

    # ---- time comparisons ----

    def _run_time_comparisons(self, df, column_types):
        date_cols = self.detect_time_comparisons(column_types)
        _, numeric_cols = self.detect_group_comparisons(column_types)
        results = []

        for date_col in date_cols:
            for value_col in numeric_cols[:3]:  # cap to avoid explosion
                result = self.compare_time_periods(df, date_col, value_col, period="year")
                if result:
                    results.append(result)
        return results

    def compare_time_periods(self, df, date_col, value_col, period="year"):
        temp = df[[date_col, value_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna()
        if temp.empty:
            return None

        if period == "year":
            temp["period"] = temp[date_col].dt.year
        elif period == "month":
            temp["period"] = temp[date_col].dt.to_period("M").astype(str)
        else:
            raise ValueError(f"Unsupported period: {period}")

        groups = {
            str(name): g[value_col].dropna()
            for name, g in temp.groupby("period")
            if len(g[value_col].dropna()) >= MIN_GROUP_SIZE
        }
        if len(groups) < 2:
            return None

        period_stats = {
            name: {
                "count": int(len(vals)),
                "mean": round(float(vals.mean()), 4),
            }
            for name, vals in groups.items()
        }

        value_lists = list(groups.values())
        if len(value_lists) == 2:
            stat, p_value = stats.ttest_ind(value_lists[0], value_lists[1], equal_var=False)
            test_name = "welch_t_test"
        else:
            stat, p_value = stats.f_oneway(*value_lists)
            test_name = "anova"

        effect_size_type, effect_size, mean_diff = self._effect_size(value_lists)

        means = [s["mean"] for s in period_stats.values()]
        trend = "increasing" if means[-1] > means[0] else "decreasing" if means[-1] < means[0] else "flat"

        return {
            "comparison_type": "time",
            "date_column": date_col,
            "value_column": value_col,
            "period_granularity": period,
            "period_stats": period_stats,
            "trend": trend,
            "test": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "is_significant": bool(p_value < 0.05),
            "effect_size_type": effect_size_type,
            "effect_size": effect_size,
            "mean_difference": mean_diff,
        }

    # ---- AI interpretation ----

    def _generate_ai_insight(self, result):
        """Turn a raw statistical result into a short business-facing interpretation."""
        if self.llm is None:
            return self._fallback_insight(result)

        try:
            subject = result.get("group_column") or result.get("date_column")
            stats_block = result.get("group_stats") or result.get("period_stats")
            prompt = f"""
You are a senior data analyst writing for a business audience.

Comparison type: {result["comparison_type"]}
Column compared: {subject}
Value column: {result["value_column"]}
Statistical test used: {result["test"]}
Test statistic: {result["statistic"]}
p-value: {result["p_value"]}
Statistically significant (p < 0.05): {result["is_significant"]}
Effect size: {result["effect_size_type"]} = {result["effect_size"]}
Mean difference / range: {result["mean_difference"]}
Per-group/period stats: {stats_block}

Write a plain-language summary of what this comparison shows, why it
matters for the business, and one concrete recommendation. Do not just
restate the numbers — interpret them. If the result is not significant
or the effect size is small, say so plainly rather than overstating it.
"""
            structured_llm = self.llm.with_structured_output(ComparisonInsight)
            insight: ComparisonInsight = structured_llm.invoke(prompt)
            return insight.model_dump()
        except Exception as e:
            logger.error("LLM error generating comparison insight: %s", e)
            return self._fallback_insight(result)

    def _fallback_insight(self, result):
        sig_phrase = "a statistically significant" if result["is_significant"] else "no statistically significant"
        subject = result.get("group_column") or result.get("date_column")
        return {
            "summary": (
                f"There is {sig_phrase} difference in {result['value_column']} "
                f"across {subject} (p = {result['p_value']}, "
                f"{result['effect_size_type']} = {result['effect_size']})."
            ),
            "business_impact": "Review the underlying data before making decisions based on this result.",
            "recommendation": "Investigate further with domain context before acting on this finding.",
        }