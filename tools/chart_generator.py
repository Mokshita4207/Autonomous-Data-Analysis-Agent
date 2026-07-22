import os
import logging
from typing import Optional
from pydantic import BaseModel, Field

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_CHART_TYPES = ["histogram", "bar_chart", "line_chart", "scatter_plot", "box_plot", "heatmap"]
ALLOWED_AGGREGATIONS = ["mean", "sum", "count", "median", "min", "max"]

# Deterministic fallback map — same logic as the earlier rule-based
# ChartGenerator, used only if the LLM path fails.
FALLBACK_CHART_MAP = {
    "univariate_numeric": "histogram",
    "univariate_categorical": "bar_chart",
    "bivariate_numeric_numeric": "scatter_plot",
    "trend_over_time": "line_chart",
    "correlation_matrix": "heatmap",
    "anomaly_detection": "scatter_plot",
    "data_quality": "bar_chart",
    "general_profiling": "histogram",
    "target_relationship": "box_plot",
    "segment_comparison": "box_plot",
    "bivariate_categorical_numeric": "box_plot",
}


class ChartRecommendation(BaseModel):
    chart_type: str = Field(description=f"One of: {ALLOWED_CHART_TYPES}")
    x_axis: str = Field(description="Column name for the x-axis, or '' if not applicable")
    y_axis: str = Field(description="Column name for the y-axis, or '' if not applicable")
    color_by: str = Field(default="", description="Optional column name to color/group by, or '' if not applicable")
    aggregation: str = Field(
        default="",
        description=f"Aggregation to apply when grouping (one of: {ALLOWED_AGGREGATIONS}), or '' if not applicable",
    )
    title: str = Field(description="Short chart title")
    reasoning: str = Field(description="One sentence explaining the choice")


class ChartGeneratorAgent:

    def __init__(self, output_dir="outputs/charts", llm=None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm = llm or self._default_llm()

    def _default_llm(self):
        try:
            from dotenv import load_dotenv
            from langchain_google_genai import ChatGoogleGenerativeAI
            load_dotenv()
            return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        except Exception:
            return None

    def run(self, df, plan, column_types, dataset_name="dataset"):
        """
        df: raw/cleaned DataFrame
        plan: list of AnalysisTask dicts from AnalysisPlannerAgent
        column_types: dict from ColumnClassifier.classify()
        """
        results = []
        for task in plan:
            try:
                if not task["columns"]:
                    continue
                recommendation = self.recommend_chart(df, task, column_types)
                path = self.generate_chart(df, task, recommendation, dataset_name)
                results.append({
                    "analysis_id": task["analysis_id"],
                    "chart_type": recommendation.chart_type,
                    "chart_path": path,
                    "reasoning": recommendation.reasoning,
                })
            except Exception as e:
                results.append({
                    "analysis_id": task["analysis_id"],
                    "chart_type": None,
                    "chart_path": None,
                    "error": str(e),
                })
        return results

    def _build_columns_summary(self, df, task, column_types):
        """
        Build a richer per-column summary for the LLM prompt: type, role/subtype
        (if provided upstream by the column classifier), unique value count, and
        percentage of missing values. Gives the LLM enough context to make
        smarter chart, axis, color, and aggregation choices.
        """
        summary = {}
        n_rows = len(df) if df is not None else 0

        for col in task["columns"]:
            raw_meta = column_types.get(col)
            if isinstance(raw_meta, dict):
                col_type = raw_meta.get("type")
                role = raw_meta.get("role_hint", "")
                subtype = raw_meta.get("subtype", "")
            else:
                col_type = raw_meta
                role = ""
                subtype = ""

            entry = {"type": col_type}
            if role:
                entry["role"] = role
            if subtype:
                entry["subtype"] = subtype

            if df is not None and col in df.columns:
                series = df[col]
                missing_pct = round(series.isna().mean() * 100, 2) if n_rows else 0.0
                entry["unique_values"] = int(series.nunique(dropna=True))
                entry["missing_pct"] = missing_pct

            summary[col] = entry

        return summary

    def recommend_chart(self, df, task, column_types):
        """Understand Analysis -> Recommend Chart (LLM or fallback)."""
        if self.llm is None:
            return self._recommend_chart_fallback(task)

        try:
            columns_summary = self._build_columns_summary(df, task, column_types)
            prompt = f"""
You are a data visualization expert agent.

Analysis type: {task["analysis_type"]}
Columns involved and their details (type, role/subtype if known, number of
unique values, and % missing): {columns_summary}
Rationale for this analysis: {task["rationale"]}

Choose the single best chart type from this list only: {ALLOWED_CHART_TYPES}
Pick appropriate x_axis and y_axis column names from the columns involved.
Optionally pick a color_by column to group/segment the chart by (leave '' if
not useful).
If the chart requires grouping numeric values by a categorical column, choose
an aggregation from this list only: {ALLOWED_AGGREGATIONS} (leave '' if no
aggregation is needed, e.g. for scatter plots or heatmaps).
"""
            structured_llm = self.llm.with_structured_output(ChartRecommendation)
            result: ChartRecommendation = structured_llm.invoke(prompt)

            if result.chart_type not in ALLOWED_CHART_TYPES:
                raise ValueError(f"LLM returned disallowed chart_type: {result.chart_type}")

            if result.aggregation and result.aggregation not in ALLOWED_AGGREGATIONS:
                raise ValueError(f"LLM returned disallowed aggregation: {result.aggregation}")

            logger.info("Chart recommendation for %s: %s (%s)",
                        task["analysis_id"], result.chart_type, result.reasoning)
            return result
        except Exception as e:
            logger.error("LLM error during chart recommendation: %s", e)
            return self._recommend_chart_fallback(task)

    def _recommend_chart_fallback(self, task):
        columns = task["columns"]
        chart_type = FALLBACK_CHART_MAP.get(task["analysis_type"], "histogram")
        return ChartRecommendation(
            chart_type=chart_type,
            x_axis=columns[0] if columns else "",
            y_axis=columns[1] if len(columns) > 1 else "",
            color_by="",
            aggregation="mean" if chart_type == "bar_chart" else "",
            title=task["rationale"],
            reasoning="Rule-based fallback (LLM unavailable or failed).",
        )

    def generate_chart(self, df, task, recommendation, dataset_name):
        """Generate Chart -> Save Chart (interactive Plotly HTML)."""
        analysis_id = task["analysis_id"]
        chart_type = recommendation.chart_type
        x, y = recommendation.x_axis, recommendation.y_axis
        color = recommendation.color_by or None
        agg = recommendation.aggregation or "mean"
        title = recommendation.title

        filename = f"{dataset_name}_{analysis_id}_{chart_type}.html"
        path = os.path.join(self.output_dir, filename)

        if color and color not in df.columns:
            color = None

        if chart_type == "histogram":
            col = x or task["columns"][0]
            fig = px.histogram(
                df, x=col, color=color, nbins=20,
                title=title or f"Distribution of {col}",
            )

        elif chart_type == "bar_chart":
            if y and pd.api.types.is_numeric_dtype(df[y]):
                group_cols = [x] + ([color] if color else [])
                grouped = (
                    df.groupby(group_cols, dropna=False)
                    .agg(value=(y, agg))
                    .reset_index()
                )

                fig = px.bar(
                    grouped,
                    x=x,
                    y="value",
                    color=color,
                    title=title or f"{agg.title()} of {y} by {x}"
                )
            elif y:
                cross = pd.crosstab(df[x], df[y]).reset_index().melt(id_vars=x, var_name=y, value_name="count")
                fig = px.bar(
                    cross, x=x, y="count", color=y,
                    title=title or f"{x} vs {y} breakdown",
                )
            else:
                counts = df[x].value_counts().reset_index()
                counts.columns = [x, "count"]
                fig = px.bar(
                    counts, x=x, y="count",
                    title=title or f"{x} breakdown",
                )

        elif chart_type == "scatter_plot":
            fig = px.scatter(
                df, x=x, y=y, color=color, opacity=0.6,
                title=title or f"{y} vs {x}",
            )

        elif chart_type == "box_plot":

            if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
                num_col = y
                group_col = x
            else:
                num_col = x
                group_col = y

            fig = px.box(
                df,
                x=group_col,
                y=num_col,
                color=color,
                title=title or f"{num_col} across {group_col}",
    )

        elif chart_type == "line_chart":
            temp = df[[x, y] + ([color] if color else [])].copy()
            temp[x] = pd.to_datetime(temp[x], errors="coerce")
            temp = temp.dropna(subset=[x, y]).sort_values(x)
            fig = px.line(
                temp, x=x, y=y, color=color,
                title=title or f"{y} over {x}",
            )

        elif chart_type == "heatmap":
            corr = df[task["columns"]].corr(numeric_only=True)
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                )
            )
            fig.update_layout(title=title or "Correlation matrix")

        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        fig.update_layout(template="plotly_white")
        fig.write_html(path, include_plotlyjs="cdn")
        return path