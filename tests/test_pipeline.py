from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler
from tools.column_classifier import ColumnClassifier

from agents.analysis_planner_agent import AnalysisPlannerAgent

from tools.code_generator import CodeGeneratorAgent
from tools.code_executor import CodeExecutor
from agents.self_corrector import SelfCorrector


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

    # Generate Python code
    generated_code = generator.generate_code(
        task
    )

    if generated_code["code"] is None:
        print("Code Generation Failed:")
        print(generated_code["error"])
        continue
    # Execute generated code
    execution = executor.run(
        generated_code,
        df
    )

    # Retry if execution fails
    if execution["status"] == "Failed":

        print("\nExecution Failed")
        print(execution["error"])

        execution = corrector.run(
            generated_code,
            df
        )

    print("\nExecution Result:\n")
    print(execution)