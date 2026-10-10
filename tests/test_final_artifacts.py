from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "reports" / "final_artifacts"


def test_required_model_checkpoints_exist():
    checkpoints = [
        "deep_learning/cnn_baseline/best_lightweight_mel_cnn.pt",
        "deep_learning/cnn_lstm/best_cnn_lstm.pt",
        "deep_learning/multifeature_cnn/best_multifeature_cnn.pt",
        "deep_learning/multifeature_cnn_12b/best_regularized_multifeature_cnn.pt",
    ]
    for relative_path in checkpoints:
        path = PROCESSED / relative_path
        assert path.is_file(), f"Missing checkpoint: {path}"
        assert path.stat().st_size > 0, f"Empty checkpoint: {path}"


def test_deep_learning_metric_files_are_readable():
    metric_files = [
        "deep_learning/cnn_baseline/cnn_baseline_metrics.csv",
        "deep_learning/cnn_lstm/cnn_lstm_metrics.csv",
        "deep_learning/multifeature_cnn/multifeature_cnn_metrics.csv",
        "deep_learning/multifeature_cnn_12b/metrics.csv",
    ]
    required = {
        "accuracy", "balanced_accuracy", "macro_precision",
        "macro_recall", "macro_f1", "weighted_f1",
    }
    for relative_path in metric_files:
        path = PROCESSED / relative_path
        assert path.is_file(), f"Missing metric file: {path}"
        df = pd.read_csv(path)
        assert not df.empty, f"Metric file is empty: {path}"
        assert "model" in df.columns
        assert required.issubset(df.columns), (
            f"Missing metric columns in {path}: {required - set(df.columns)}"
        )


def test_metric_values_are_in_valid_ranges():
    metric_files = [
        "deep_learning/cnn_baseline/cnn_baseline_metrics.csv",
        "deep_learning/cnn_lstm/cnn_lstm_metrics.csv",
        "deep_learning/multifeature_cnn/multifeature_cnn_metrics.csv",
        "deep_learning/multifeature_cnn_12b/metrics.csv",
    ]
    columns = [
        "accuracy", "balanced_accuracy", "macro_precision",
        "macro_recall", "macro_f1", "weighted_f1",
    ]
    for relative_path in metric_files:
        df = pd.read_csv(PROCESSED / relative_path)
        for column in columns:
            values = pd.to_numeric(df[column], errors="coerce")
            assert values.notna().all(), f"Invalid values in {relative_path}: {column}"
            assert values.between(0, 1).all(), (
                f"Values outside [0, 1] in {relative_path}: {column}"
            )


def test_deep_learning_prediction_files_are_readable():
    prediction_files = [
        "deep_learning/cnn_baseline/cnn_test_predictions.csv",
        "deep_learning/cnn_lstm/cnn_lstm_test_predictions.csv",
        "deep_learning/multifeature_cnn/multifeature_cnn_test_predictions.csv",
        "deep_learning/multifeature_cnn_12b/test_predictions.csv",
    ]
    for relative_path in prediction_files:
        path = PROCESSED / relative_path
        assert path.is_file(), f"Missing predictions: {path}"
        df = pd.read_csv(path)
        assert not df.empty, f"Empty predictions: {path}"
        assert len(df.columns) > 1


def test_consolidated_comparison_exists_if_generated():
    comparison = ARTIFACTS / "cycle_level_model_comparison.csv"
    if comparison.exists():
        df = pd.read_csv(comparison)
        assert not df.empty
        assert "model" in df.columns
        assert "model_family" in df.columns


def test_patient_level_results_are_separate():
    patient_file = (
        PROCESSED / "deep_learning" / "patient_level_prediction"
        / "patient_aggregation_comparison.csv"
    )
    assert patient_file.is_file(), f"Missing patient-level comparison: {patient_file}"
    df = pd.read_csv(patient_file)
    assert not df.empty
    assert "aggregation_method" in df.columns
    assert "accuracy" in df.columns
    assert "balanced_accuracy" in df.columns
