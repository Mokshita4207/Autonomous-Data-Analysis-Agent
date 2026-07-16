from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler
from tools.column_classifier import ColumnClassifier
from agents.analysis_planner_agent import AnalysisPlannerAgent
from tools.code_generator import CodeGeneratorAgent
from tools.code_executor import CodeExecutor
from agents.self_corrector import SelfCorrector


def main():

    loader = DataLoader()
    df = loader.load_file("data/train.csv")

    profiler = DataProfiler()
    profile = profiler.run(df)

    classifier = ColumnClassifier()
    column_types = classifier.classify(df)

    planner = AnalysisPlannerAgent()
    tasks = planner.run(column_types, profile)

    generator = CodeGeneratorAgent()

    executor = CodeExecutor()

    corrector = SelfCorrector()

    for task in tasks:

        print("\n" + "=" * 70)
        print(f"Analysis : {task['analysis_type']}")
        print("=" * 70)

        generated_code = generator.generate_code(task)

        execution = executor.run(
            generated_code,
            df
        )

        if execution["status"] == "Failed":

            execution = corrector.run(
                generated_code,
                df
            )

        print("\nExecution Result:\n")
        print(execution)


if __name__ == "__main__":
    main()