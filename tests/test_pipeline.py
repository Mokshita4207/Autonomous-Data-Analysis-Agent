from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler
from tools.column_classifier import ColumnClassifier

from agents.analysis_planner_agent import AnalysisPlannerAgent

from tools.code_generator import CodeGeneratorAgent
from tools.code_executor import CodeExecutor
from agents.self_corrector import SelfCorrector

from tools.chart_generator import ChartGeneratorAgent
from agents.insight_extractor import InsightExtractor

from tools.anomaly_detector import AnomalyDetector
from tools.statistical_analyzer import StatisticalAnalyzer


# -----------------------------
# Load Dataset
# -----------------------------
loader = DataLoader()

df = loader.load_file(
    "data/heart-disease.csv"
)


# -----------------------------
# Week 1 Components
# -----------------------------
profiler = DataProfiler()
profile = profiler.run(df)

classifier = ColumnClassifier()
column_types = classifier.classify(df)


# -----------------------------
# Week 2 Components
# -----------------------------
planner = AnalysisPlannerAgent()
generator = CodeGeneratorAgent()
executor = CodeExecutor()
corrector = SelfCorrector()


# -----------------------------
# Week 3 Components
# -----------------------------
chart_generator = ChartGeneratorAgent()
insight_extractor = InsightExtractor()
anomaly_detector = AnomalyDetector()
statistical_analyzer = StatisticalAnalyzer()


# -----------------------------
# Create Analysis Plan
# -----------------------------
tasks = planner.run(
    column_types,
    profile
)


# -----------------------------
# Execute Every Analysis
# -----------------------------
for task in tasks:

    print("\n" + "=" * 70)
    print(f"Analysis : {task['analysis_type']}")
    print("=" * 70)

    # -----------------------------
    # Generate Python code
    # -----------------------------
    generated_code = generator.generate_code(task)

    if generated_code["code"] is None:
        print("Code Generation Failed:")
        print(generated_code["error"])
        continue

    # -----------------------------
    # Execute generated code
    # -----------------------------
    execution = executor.run(
        generated_code,
        df
    )

    # -----------------------------
    # Retry if execution fails
    # -----------------------------
    if execution["status"] == "Failed":

        print("\nExecution Failed")
        print(execution["error"])

        execution = corrector.run(
            generated_code,
            df
        )

    print("\nExecution Result:\n")
    print(execution)

    # -----------------------------
    # Generate Chart
    # -----------------------------

    execution["analysis_id"] = task["analysis_id"]
    execution["analysis_type"] = task["analysis_type"]
    
    chart = chart_generator.run(
        df=df,
        plan=[task],
        column_types=column_types,
        dataset_name="heart_disease"
    )

    print("\nChart\n")
    print(chart)

    # -----------------------------
    # Extract Insights
    # -----------------------------
    insights = insight_extractor.run(execution)

    print("\nInsights\n")
    print(insights)


# ===================================================
# Dataset-level Analysis (Run Once)
# ===================================================

# -----------------------------
# Statistical Analysis
# -----------------------------
print("\nRunning Statistical Analysis...\n")

statistics = statistical_analyzer.correlation_analysis(
    df,
    method="pearson"
)

print(statistics)


# -----------------------------
# Anomaly Detection
# -----------------------------
print("\nRunning Anomaly Detection...\n")

anomalies = anomaly_detector.flag_outliers(
    df,
    method="isolation_forest"      # or "zscore"
)

print(anomalies)