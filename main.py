from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler
from tools.column_classifier import ColumnClassifier
from agents.analysis_planner_agent import AnalysisPlannerAgent
from tools.code_generator import CodeGeneratorAgent
from tools.code_executor import CodeExecutor
from agents.self_corrector import SelfCorrector
from tools.statistical_analyzer import StatisticalAnalyzer
from tools.anomaly_detector import AnomalyDetector


def main():

    loader = DataLoader()

    df = loader.load_file(
        "data/train.csv"
    )

    profiler = DataProfiler()

    profile = profiler.run(
        df
    )

    classifier = ColumnClassifier()

    column_types = classifier.classify(
        df
    )

    planner = AnalysisPlannerAgent()

    tasks = planner.run(
        column_types,
        profile
    )

    generator = CodeGeneratorAgent()

    executor = CodeExecutor()

    corrector = SelfCorrector()

    for task in tasks:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Analysis : "
            f"{task['analysis_type']}"
        )

        print(
            "=" * 70
        )

        generated_code = generator.generate_code(
            task
        )

        execution = executor.run(
            generated_code,
            df
        )

        if execution["status"] == "Failed":

            execution = corrector.run(
                generated_code,
                df
            )

        print(
            "\nExecution Result:\n"
        )

        print(
            execution
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATISTICAL ANALYSIS"
    )

    print(
        "=" * 70
    )

    statistical_analyzer = StatisticalAnalyzer()

    statistical_result = (
        statistical_analyzer.correlation_analysis(
            df
        )
    )

    print(
        "\nCorrelation Matrix:\n"
    )

    print(
        statistical_result["correlation"]
    )

    print(
        "\nSignificant Correlations:\n"
    )

    for correlation in (
        statistical_result["significance"]
    ):

        print(
            correlation
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANOMALY DETECTION"
    )

    print(
        "=" * 70
    )

    anomaly_detector = AnomalyDetector()

    zscore_result = (
        anomaly_detector.detect_zscore(
            df
        )
    )

    isolation_forest_result = (
        anomaly_detector.detect_isolation_forest(
            df
        )
    )

    print(
        "\nZ-score Anomaly Detection:\n"
    )

    print(
        f"Outliers: "
        f"{zscore_result['outlier_count']}"
    )

    print(
        f"Anomaly Percentage: "
        f"{zscore_result['anomaly_percentage']:.2f}%"
    )

    print(
        "\nIsolation Forest Anomaly Detection:\n"
    )

    print(
        f"Outliers: "
        f"{isolation_forest_result['outlier_count']}"
    )

    print(
        f"Anomaly Percentage: "
        f"{isolation_forest_result['anomaly_percentage']:.2f}%"
    )

    member_2_result = {

        "correlation":
            statistical_result[
                "correlation"
            ],

        "pvalue":
            statistical_result[
                "pvalue"
            ],

        "significance":
            statistical_result[
                "significance"
            ],

        "anomalies": {

            "zscore":
                zscore_result,

            "isolation_forest":
                isolation_forest_result

        }

    }

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MEMBER 2 FINAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        member_2_result
    )


if __name__ == "__main__":

    main()