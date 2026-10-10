from __future__ import annotations
import json, tempfile, time, os, logging
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from .services.models import (
    ROOT,
    MODEL_SPECS,
    load_signal,
    load_dl_model,
    predict_dl,
    explain_dl,
    extract_features,
    CLASS_NAMES,
)
from .services.audio_processing import (
    decode_audio,
    extract_annotated_cycle,
    fixed_length_waveform,
    parse_cycle_annotations,
    read_native_mono,
    resample_peak_normalize,
    split_recording_into_windows,
)
from .services.classical import (
    artifacts,
    explain_classical,
    extract_all_features,
    predict_classical,
)
from .services.classical import DIR as CLASSICAL_ARTIFACT_DIR
from .schemas import PredictionResponse

app = FastAPI(
    title="Respiratory Sound Research API",
    version="1.0.0",
    description="Local academic research interface; predictions are not diagnoses.",
)
configured_origins = [
    "https://respiratory-disease-classification-seven.vercel.app",
    *[
        origin.strip().rstrip("/")
        for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
        if origin.strip()
    ],
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *configured_origins,
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}):5173",
    allow_methods=["*"],
    allow_headers=["*"],
)
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
MAX_DURATION_SECONDS = float(os.getenv("MAX_AUDIO_DURATION_SECONDS", 120))
logger = logging.getLogger(__name__)
MODEL_META = {
    "logistic_regression": ("Logistic Regression", "traditional_ml"),
    "svm": ("SVM", "traditional_ml"),
    "random_forest": ("Random Forest", "traditional_ml"),
    "lightweight_mel_cnn": ("Lightweight Mel CNN", "deep_learning"),
    "multifeature_cnn": ("Multi-Feature CNN", "deep_learning"),
    "regularized_multifeature_cnn": (
        "Regularized Multi-Feature CNN (12B)",
        "deep_learning",
    ),
    "cnn_lstm": ("CNN-LSTM", "deep_learning"),
}


def csv(path):
    p = ROOT / path
    if not p.is_file():
        return []
    try:
        return json.loads(pd.read_csv(p).to_json(orient="records"))
    except Exception:
        return []


def json_file(path):
    p = ROOT / path
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def public_model(mid):
    name, family = MODEL_META[mid]
    reasons = []
    checked = False
    artifact_version = ""
    artifact_size_bytes = None
    try:
        if family == "deep_learning":
            load_dl_model(mid)
            checkpoint = ROOT / "models" / MODEL_SPECS[mid][1]
            artifact_version = "patient-disjoint checkpoint"
            artifact_size_bytes = checkpoint.stat().st_size
        else:
            metadata, features, scaler, encoder, estimators = artifacts()
            checked = mid in estimators
            artifact_version = metadata.get("package_version", "legacy-official-split")
            artifact_size_bytes = (CLASSICAL_ARTIFACT_DIR / f"{mid}.joblib").stat().st_size
        checked = True
    except Exception as e:
        reasons.append(str(e))
    return {
        "model_id": mid,
        "display_name": name,
        "family": family,
        "available": checked,
        "artifact_validation": "loaded strictly" if checked else "failed",
        "artifact_version": artifact_version or None,
        "artifact_size_bytes": artifact_size_bytes,
        "input_contract": "WAV input; optional ICBHI-style cycle annotations, otherwise consecutive five-second windows; 4 kHz, mono, peak-normalized",
        "class_names": CLASS_NAMES,
        "reason": None
        if checked
        else (reasons[0] if reasons else "required artifact unavailable"),
    }


def saved_model_inventory():
    """Check artifact paths without importing estimators or loading checkpoints."""
    inventory = []
    for model_id, (display_name, family) in MODEL_META.items():
        if family == "deep_learning":
            checkpoint = ROOT / "models" / MODEL_SPECS[model_id][1]
            normalization = ROOT / "models" / MODEL_SPECS[model_id][2]
            present = checkpoint.is_file() and normalization.is_file()
            artifact = checkpoint
        else:
            artifact = CLASSICAL_ARTIFACT_DIR / f"{model_id}.joblib"
            present = artifact.is_file()
        inventory.append(
            {
                "model_id": model_id,
                "model_name": display_name,
                "display_name": display_name,
                "family": family,
                "artifact_exists": present,
                "artifact_size_bytes": artifact.stat().st_size if present else None,
            }
        )
    return inventory


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "respiratory-research-api"}


@app.get("/api/models")
def models():
    return {"models": [public_model(m) for m in MODEL_META]}


@app.get("/api/overview")
def overview():
    pats = pd.read_csv(ROOT / "data/processed/patients_clean.csv")
    rec = pd.read_csv(ROOT / "data/processed/recordings_clean.csv")
    cycles = pd.read_csv(ROOT / "data/processed/cycles_clean.csv")
    return {
        "title": "Respiratory Disease Classification",
        "description": "Research platform for lung-sound data processing, classification, evaluation, and interpretation.",
        "dataset": {
            "patients": int(pats.patient_id.nunique()),
            "recordings": int(rec.wav_path.nunique()),
            "cycles": int(len(cycles)),
            "classes": pats.diagnosis.value_counts()
            .rename_axis("diagnosis")
            .reset_index(name="count")
            .to_dict("records"),
        },
        # This is a saved artifact inventory. Loading and validating model weights is
        # deferred until the user opens Predict Audio.
        "models": saved_model_inventory(),
        "highlights": csv("reports/final_artifacts/cycle_level_model_comparison.csv"),
        "disclaimer": "For academic research only. Model outputs are not medical diagnoses.",
    }


@app.get("/api/eda/summary")
def eda():
    return {
        "disease_distribution": csv("data/processed/eda/disease_distribution.csv"),
        "sound_distribution": csv("data/processed/eda/sound_distribution.csv"),
        "patient_audio_summary": csv("data/processed/eda/patient_audio_summary.csv"),
        "disease_sound_counts": csv("data/processed/eda/disease_sound_counts.csv"),
        "cleaning_summary": csv("data/processed/cleaning_summary.csv"),
        "patient_count": int(
            pd.read_csv(ROOT / "data/processed/patients_clean.csv").patient_id.nunique()
        ),
        "recording_count": int(
            pd.read_csv(
                ROOT / "data/processed/recordings_clean.csv"
            ).wav_path.nunique()
        ),
        "cycle_count": len(pd.read_csv(ROOT / "data/processed/cycles_clean.csv")),
        "pca": csv("data/processed/pca_clustering/pca_explained_variance.csv"),
        "clusters": csv("data/processed/pca_clustering/kmeans_cluster_analysis.csv"),
    }


@app.get("/api/features/summary")
def features():
    return {
        "selected_features": csv(
            "data/processed/feature_selection/selected_features.csv"
        ),
        "ranking": csv(
            "data/processed/feature_selection/final_feature_selection_ranking.csv"
        ),
        "statistics": csv("data/processed/analysis/feature_statistics.csv"),
        "variance": csv("data/processed/analysis/feature_variance.csv"),
        "pca": csv("data/processed/pca_clustering/pca_explained_variance.csv"),
        "clusters": csv("data/processed/pca_clustering/kmeans_cluster_analysis.csv"),
    }


@app.get("/api/pipeline/summary")
def pipeline_summary():
    """Read-only summaries produced by notebooks 02–08."""
    return {
        "cleaning_summary": csv("data/processed/cleaning_summary.csv"),
        "feature_statistics": csv("data/processed/analysis/feature_statistics.csv"),
        "feature_variance": csv("data/processed/analysis/feature_variance.csv"),
        "high_correlations": csv("data/processed/analysis/high_correlation_pairs.csv"),
        "disease_feature_means": csv("data/processed/analysis/disease_feature_means.csv"),
        "sound_feature_means": csv("data/processed/analysis/sound_feature_means.csv"),
        "anova_ranking": csv("data/processed/feature_selection/anova_feature_scores.csv"),
        "mutual_information": csv("data/processed/feature_selection/mutual_information_scores.csv"),
        "random_forest_importance": csv("data/processed/feature_selection/random_forest_feature_importance.csv"),
    }


@app.get("/api/development/summary")
def development_summary():
    """Saved training histories/metrics only; does not instantiate a model."""
    experiments = {
        "lightweight_mel_cnn": ("cnn_baseline", "cnn_training_history.csv", "cnn_baseline_metrics.csv"),
        "multifeature_cnn": ("multifeature_cnn", "multifeature_cnn_training_history.csv", "multifeature_cnn_metrics.csv"),
        "regularized_multifeature_cnn_12b": ("multifeature_cnn_12b", "training_history.csv", "metrics.csv"),
        "cnn_lstm": ("cnn_lstm", "cnn_lstm_training_history.csv", "cnn_lstm_metrics.csv"),
    }
    result = {}
    for key, (directory, history, metrics) in experiments.items():
        root = f"data/processed/deep_learning/{directory}"
        result[key] = {
            "history": csv(f"{root}/{history}"),
            "metrics": csv(f"{root}/{metrics}"),
        }
    result["classical_metrics"] = csv("data/processed/classical_ml/classical_ml_model_comparison.csv")
    result["classical_per_disease"] = csv("data/processed/classical_ml/classical_ml_per_disease_results.csv")
    return result


@app.get("/api/clustering/summary")
def clustering():
    return {
        "pca": csv("data/processed/pca_clustering/pca_explained_variance.csv"),
        "clusters": csv("data/processed/pca_clustering/kmeans_cluster_analysis.csv"),
    }


def cnn_baseline_confusion():
    saved_matrix = (
        ROOT
        / "data/processed/deep_learning/cnn_baseline/cnn_baseline_confusion_matrix.csv"
    )
    if saved_matrix.is_file():
        return csv(
            "data/processed/deep_learning/cnn_baseline/cnn_baseline_confusion_matrix.csv"
        )

    predictions = csv(
        "data/processed/deep_learning/cnn_baseline/cnn_test_predictions.csv"
    )
    if not predictions:
        return []

    matrix = {
        label: {prediction: 0 for prediction in CLASS_NAMES} for label in CLASS_NAMES
    }
    for row in predictions:
        actual = row.get("true_disease")
        predicted = row.get("predicted_disease")
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    return [{"true_class": label, **matrix[label]} for label in CLASS_NAMES]


@app.get("/api/evaluation/models")
def evaluations():
    return {
        "cycle_level": csv("reports/final_artifacts/cycle_level_model_comparison.csv"),
        "common_patient_split": {
            "cycle_level": csv("reports/research_audit/common_patient_split/overall_metrics.csv"),
            "cycle_per_class": csv("reports/research_audit/common_patient_split/per_class_metrics.csv"),
            "patient_level": csv("reports/research_audit/common_patient_split/patient_level_overall_metrics.csv"),
            "patient_per_class": csv("reports/research_audit/common_patient_split/patient_level_per_class_metrics.csv"),
            "protocol": json_file("reports/research_audit/common_patient_split/protocol.json"),
            "confusion": {
                model_id: csv(f"reports/research_audit/common_patient_split/confusion_matrix_{model_id}.csv")
                for model_id in MODEL_META
            },
        },
        "classical": csv(
            "data/processed/classical_ml/classical_ml_per_disease_results.csv"
        ),
        "confusion": {
            "cnn_baseline": cnn_baseline_confusion(),
            **{
                k: csv(f"data/processed/deep_learning/{k}/{f}")
                for k, f in [
                    ("cnn_lstm", "cnn_lstm_confusion_matrix.csv"),
                    ("multifeature_cnn", "multifeature_cnn_confusion_matrix.csv"),
                    ("multifeature_cnn_12b", "confusion_matrix.csv"),
                ]
            },
        },
    }


@app.get("/api/evaluation/cycle-level")
def cycle_eval():
    return csv("reports/final_artifacts/cycle_level_model_comparison.csv")


@app.get("/api/evaluation/patient-level")
def patient_eval():
    base = ROOT / "data/processed/deep_learning/patient_level_prediction"

    def matrix(filename):
        frame = pd.read_csv(base / filename, index_col=0)
        frame.index.name = "true_class"
        return json.loads(frame.reset_index().to_json(orient="records"))

    majority = pd.read_csv(base / "patient_majority_predictions.csv")
    mean = pd.read_csv(base / "patient_probability_predictions.csv")
    same = majority[["patient_id", "predicted_disease"]].merge(
        mean[["patient_id", "predicted_disease"]],
        on="patient_id",
        suffixes=("_majority", "_mean"),
    )
    identical = int(
        (same.predicted_disease_majority == same.predicted_disease_mean).sum()
    )
    return {
        "metrics": csv("reports/final_artifacts/patient_level_model_comparison.csv"),
        "majority_confusion": matrix("majority_vote_confusion_matrix.csv"),
        "mean_confusion": matrix("mean_probability_confusion_matrix.csv"),
        "aggregation_diagnostics": {
            "test_patients": int(len(same)),
            "matching_predicted_classes": identical,
            "predictions_identical": identical == len(same),
            "explanation": "Both aggregation methods produced the same winning class for every saved test patient; their reported metrics and confusion matrices therefore match."
            if identical == len(same)
            else "The two saved aggregation methods differ on at least one patient.",
        },
    }


@app.get("/api/xai/summary")
def xai():
    return {
        "summary": csv("data/processed/deep_learning/xai/multi_class_xai_summary.csv"),
        "metadata": json.loads(
            (
                ROOT / "data/processed/deep_learning/xai/gradcam_example_metadata.json"
            ).read_text()
        ),
        "images": [
            p.name for p in (ROOT / "data/processed/deep_learning/xai").glob("*.png")
        ],
    }


@app.get("/api/xai/image/{name}")
def xai_image(name: str):
    p = (ROOT / "data/processed/deep_learning/xai" / name).resolve()
    base = (ROOT / "data/processed/deep_learning/xai").resolve()
    if p.parent != base or p.suffix.lower() != ".png" or not p.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/reports")
def reports():
    names = [
        "final_results_report.md",
        "model_registry.csv",
        "cycle_level_model_comparison.csv",
        "patient_level_model_comparison.csv",
        "final_project_validation.csv",
        "checkpoint_integrity_check.csv",
        "prediction_validation.csv",
        "model_package_manifest.csv",
    ]
    audit_names = [
        "README.md",
        "overall_metrics.csv",
        "per_class_metrics.csv",
        "patient_level_overall_metrics.csv",
        "patient_level_per_class_metrics.csv",
        "protocol.json",
        "model_artifact_sizes.csv",
    ]
    final_reports = [
        {"name": n, "url": "/api/reports/download/" + n}
        for n in names
        if (ROOT / "reports/final_artifacts" / n).is_file()
    ]
    audit_reports = [
        {"name": n, "url": "/api/reports/download/" + n, "group": "Common patient split audit"}
        for n in audit_names
        if (ROOT / "reports/research_audit/common_patient_split" / n).is_file()
        or (n == "README.md" and (ROOT / "reports/research_audit/README.md").is_file())
    ]
    return {"reports": final_reports + audit_reports}


@app.get("/api/reports/download/{name}")
def report_download(name: str):
    final_dir = (ROOT / "reports/final_artifacts").resolve()
    audit_dir = (ROOT / "reports/research_audit/common_patient_split").resolve()
    if name == "README.md":
        p = (ROOT / "reports/research_audit/README.md").resolve()
    else:
        candidates = [final_dir / name, audit_dir / name]
        p = next((candidate for candidate in candidates if candidate.is_file()), None)
    if p is None or not p.is_file() or p.parent not in {final_dir, audit_dir, (ROOT / "reports/research_audit").resolve()}:
        raise HTTPException(404, "Report not found")
    return FileResponse(p, filename=p.name)


@app.get("/api/system/status")
def status():
    # Informational page: report saved validation records without loading weights.
    records = csv("reports/final_artifacts/final_project_validation.csv")
    return {
        "validation_records": records,
        "artifact_registry": saved_model_inventory(),
        "validation_summary": json_file("reports/final_artifacts/final_project_validation.json"),
    }


async def run_prediction(file, model_ids, single_cycle_confirmed=False, annotation_file=None):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            413, f"Upload exceeds the {MAX_BYTES / (1024 * 1024):g} MiB limit"
        )
    if not file.filename or Path(file.filename).suffix.lower() != ".wav":
        raise HTTPException(415, "Upload a WAV file")
    if not model_ids:
        raise HTTPException(422, "Select at least one model")
    if len(set(model_ids)) != len(model_ids) or any(
        m not in MODEL_META for m in model_ids
    ):
        raise HTTPException(422, "Unknown or duplicate model identifier")
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            path = Path(f.name)
        info = sf.info(path)
        if info.samplerate <= 0 or info.frames <= 0:
            raise HTTPException(422, "WAV file has no valid audio frames")
        input_duration = float(info.frames / info.samplerate)
        if input_duration > MAX_DURATION_SECONDS:
            raise HTTPException(
                413, f"Audio exceeds the {MAX_DURATION_SECONDS:g}-second duration limit"
            )
        native_audio, source_sr, channels = read_native_mono(path)
        if annotation_file is not None and annotation_file.filename:
            if Path(annotation_file.filename).suffix.lower() != ".txt":
                raise HTTPException(415, "Cycle annotations must be a .txt file")
            annotation_bytes = await annotation_file.read(1024 * 1024 + 1)
            if len(annotation_bytes) > 1024 * 1024:
                raise HTTPException(413, "Annotation file exceeds the 1 MiB limit")
            try:
                annotation_text = annotation_bytes.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise HTTPException(422, "Annotation file must be UTF-8 text") from exc
            intervals = parse_cycle_annotations(annotation_text, input_duration)
            cycle_inputs = [
                (index, extract_annotated_cycle(native_audio, source_sr, start, end))
                for index, (start, end) in enumerate(intervals, start=1)
            ]
            input_mode = "annotated_recording"
            input_note = f"Extracted {len(cycle_inputs)} cycles using the uploaded annotation timestamps."
        elif single_cycle_confirmed:
            # Keep compatibility for API clients that still send the previous flag.
            cycle_inputs = [(1, resample_peak_normalize(native_audio, source_sr))]
            input_mode = "single_cycle"
            input_note = "Treated the upload as one isolated respiratory cycle (legacy API option)."
        else:
            cycle_inputs = split_recording_into_windows(native_audio, source_sr)
            input_mode = "automatic_windows"
            input_note = (
                f"Split the complete WAV into {len(cycle_inputs)} consecutive five-second "
                "windows (the final window may be shorter). These windows are not detected "
                "respiratory cycles."
            )
        cycle_count = len(cycle_inputs)

        analysis_input = fixed_length_waveform(cycle_inputs[0][1])
        analysis_features = await run_in_threadpool(extract_features, analysis_input)
        native_stride = max(1, len(native_audio) // 600)
        model_stride = max(1, len(analysis_input) // 600)
        analysis = {
            "cycle_count": cycle_count,
            "neural_input_selected": any(
                MODEL_META[model_id][1] == "deep_learning" for model_id in model_ids
            ),
            "preview_cycle_number": cycle_inputs[0][0],
            "neural_input_duration_seconds": 5.0,
            "waveform_sample_rate": 4000,
            "source_waveform_sample_rate": source_sr,
            "source_waveform_time_step_seconds": native_stride / source_sr,
            "source_waveform": native_audio[::native_stride][:600].round(5).tolist(),
            "model_input_waveform": analysis_input[::model_stride][:600].round(5).tolist(),
            "logmel": analysis_features["logmel"].round(3).tolist(),
            "mfcc": analysis_features["mfcc"].round(3).tolist(),
            "chroma": analysis_features["chroma"].round(3).tolist(),
            "processing_steps": [
                {
                    "name": "Input validation",
                    "status": "passed",
                    "detail": f"Readable WAV · {input_duration:.2f} s · {source_sr} Hz · {channels} channel(s)",
                },
                {
                    "name": "Recording segmentation",
                    "status": "passed",
                    "detail": input_note,
                },
                {
                    "name": "Segment preprocessing",
                    "status": "passed",
                    "detail": "Audio is downmixed to mono and resampled to 4 kHz. Each annotated cycle or automatic window is peak-normalized.",
                },
                {
                    "name": "Neural model input",
                    "status": "passed"
                    if any(MODEL_META[model_id][1] == "deep_learning" for model_id in model_ids)
                    else "not_selected",
                    "detail": "Selected neural models receive 20,000 samples (5 seconds), padded or truncated as needed.",
                },
                {
                    "name": "Audio feature transforms",
                    "status": "passed",
                    "detail": "Live feature views: Log-Mel (64 bands), MFCC (13 coefficients), and Chroma (12 bands).",
                },
            ],
        }

        segment_unit = "window" if input_mode == "automatic_windows" else "cycle"
        warnings = [
            "Models were trained on individual respiratory cycles.",
            f"Per-class values are mean {segment_unit} scores and are not calibrated clinical confidence.",
            "Rare diagnoses have little or no held-out patient support in the saved evaluation.",
        ]
        if input_mode == "automatic_windows":
            warnings.insert(
                1,
                "Unannotated recordings are scored in consecutive five-second windows, not detected respiratory cycles. A window may contain partial or multiple breaths, so these results are exploratory and may not match the model's cycle-based evaluation domain.",
            )
        results = []
        failures = []
        classical_ids = [mid for mid in model_ids if MODEL_META[mid][1] == "traditional_ml"]
        classical_features = []
        if classical_ids:
            for _, cycle_audio in cycle_inputs:
                classical_features.append(
                    await run_in_threadpool(extract_all_features, cycle_audio, 4000)
                )
        for mid in model_ids:
            try:
                cycle_results = []
                cycle_explanations = []
                inference_ms = 0.0
                explanation_ms = 0.0
                for index, (cycle_number, cycle_audio) in enumerate(cycle_inputs):
                    if MODEL_META[mid][1] == "deep_learning":
                        model_input = fixed_length_waveform(cycle_audio)
                        cycle_result = await run_in_threadpool(
                            predict_dl, mid, model_input
                        )
                        inference_ms += cycle_result.get("inference_ms", 0.0)
                    else:
                        cycle_result = (
                            await run_in_threadpool(
                                predict_classical,
                                cycle_audio,
                                4000,
                                [mid],
                                classical_features[index],
                            )
                        )[0]
                        inference_ms += cycle_result.get("inference_ms", 0.0)
                    # Explain one representative segment per model. Score every
                    # segment, but avoid multiplying gradient/ablation cost by
                    # the duration of a long upload.
                    if index == 0:
                        explanation_started = time.perf_counter()
                        try:
                            if MODEL_META[mid][1] == "deep_learning":
                                explanation = await run_in_threadpool(
                                    explain_dl,
                                    mid,
                                    model_input,
                                    cycle_result["predicted_class"],
                                )
                            else:
                                explanation = await run_in_threadpool(
                                    explain_classical,
                                    mid,
                                    classical_features[index],
                                    cycle_result["predicted_class"],
                                )
                            cycle_explanations.append(explanation)
                        except Exception:
                            logger.exception("Explanation failed for model %s", mid)
                        finally:
                            explanation_ms += (
                                time.perf_counter() - explanation_started
                            ) * 1000
                    cycle_results.append(
                        {
                            "cycle_number": cycle_number,
                            "predicted_class": cycle_result["predicted_class"],
                            "scores": cycle_result["scores"],
                        }
                    )

                explanation = None
                if cycle_explanations:
                    first_explanation = cycle_explanations[0]
                    if "feature_attributions" in first_explanation:
                        keys = set.intersection(
                            *[
                                set(item["feature_attributions"])
                                for item in cycle_explanations
                            ]
                        )
                        explanation = {
                            "method": first_explanation["method"],
                            "target_classes": sorted(
                                {item["target_class"] for item in cycle_explanations}
                            ),
                            "scope": f"First {segment_unit} only (segment {cycle_inputs[0][0]}); recording scores aggregate all segments",
                            "feature_attributions": {
                                key: np.mean(
                                    [
                                        item["feature_attributions"][key]
                                        for item in cycle_explanations
                                    ],
                                    axis=0,
                                )
                                .round(4)
                                .tolist()
                                for key in keys
                            },
                        }
                    else:
                        by_feature = {}
                        for item in cycle_explanations:
                            for feature in item["features"]:
                                by_feature.setdefault(feature["feature"], []).append(
                                    feature["impact"]
                                )
                        effects = [
                            {
                                "feature": name,
                                "impact": float(np.mean(values)),
                            }
                            for name, values in by_feature.items()
                        ]
                        effects.sort(
                            key=lambda item: abs(item["impact"]), reverse=True
                        )
                        explanation = {
                            "method": first_explanation["method"],
                            "target_classes": sorted(
                                {item["target_class"] for item in cycle_explanations}
                            ),
                            "score_label": first_explanation.get("score_label"),
                            "scope": f"First {segment_unit} only (segment {cycle_inputs[0][0]}); recording scores aggregate all segments",
                            "features": effects[:10],
                        }

                score_labels = {
                    label
                    for prediction in cycle_results
                    for label in (prediction["scores"] or {})
                }
                if score_labels:
                    mean_scores = {
                        label: sum(
                            (prediction["scores"] or {}).get(label, 0.0)
                            for prediction in cycle_results
                        )
                        / len(cycle_results)
                        for label in score_labels
                    }
                    predicted_class = max(mean_scores, key=mean_scores.get)
                else:
                    from collections import Counter

                    votes = Counter(
                        prediction["predicted_class"] for prediction in cycle_results
                    )
                    mean_scores = None
                    predicted_class = votes.most_common(1)[0][0]

                results.append(
                    {
                        "model_id": mid,
                        "model_name": MODEL_META[mid][0],
                        "model_family": MODEL_META[mid][1],
                        "predicted_class": predicted_class,
                        "scores": mean_scores,
                        "score_type": (
                            f"mean per-{segment_unit} score; calibration not established"
                            if mean_scores
                            else f"majority {segment_unit} vote"
                        ),
                        "inference_ms": round(inference_ms, 2),
                        "explanation_ms": round(explanation_ms, 2),
                        "warnings": warnings,
                        "cycle_predictions": cycle_results,
                        "explanation": explanation,
                    }
                )
            except Exception:
                logger.exception("Inference failed for selected model %s", mid)
                failures.append(
                    {
                        "model_id": mid,
                        "error": "This model could not process the recording. Check that the audio is valid and review backend logs.",
                    }
                )
        outcome = (
            "failure"
            if failures and not results
            else ("partial_failure" if failures else "success")
        )
        return {
            "status": outcome,
            "filename": Path(file.filename).name,
            "source_sample_rate": source_sr,
            "target_sample_rate": 4000,
            "channels": channels,
            "input_duration_seconds": round(input_duration, 3),
            "input_mode": input_mode,
            "cycle_count": cycle_count,
            "segment_count": cycle_count,
            "neural_input_duration_seconds": 5.0,
            "handling": (
                f"{input_note} Each segment is mono, resampled to 4 kHz, and peak-normalized. "
                "Neural models pad or truncate each segment to five seconds; traditional models "
                f"process the complete segment. Scores are averaged across {segment_unit}s."
            ),
            "audio_analysis": analysis,
            "warnings": warnings,
            "results": results,
            "failures": failures,
            "partial_success": bool(results and failures),
            "disclaimer": "Research output only; not a medical diagnosis.",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        logger.exception("Uploaded audio could not be processed")
        raise HTTPException(
            422,
            "Could not decode or process this WAV recording. Check that it is valid PCM audio.",
        )
    finally:
        if path:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/api/audio/preview")
async def audio_preview(file: UploadFile = File(...)):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            413, f"Upload exceeds the {MAX_BYTES / (1024 * 1024):g} MiB limit"
        )
    if not file.filename or Path(file.filename).suffix.lower() != ".wav":
        raise HTTPException(415, "Upload a WAV file")
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            path = Path(f.name)
        info = sf.info(path)
        if info.samplerate <= 0 or info.frames <= 0:
            raise HTTPException(422, "WAV file has no valid audio frames")
        duration = info.frames / info.samplerate
        if duration > MAX_DURATION_SECONDS:
            raise HTTPException(
                413, f"Audio exceeds the {MAX_DURATION_SECONDS:g}-second duration limit"
            )
        y, sr, channels = decode_audio(path, fixed_duration=False)
        from .services.models import extract_features

        feats = await run_in_threadpool(extract_features, y)
        # Bound payload size for long recordings while preserving a useful overview.
        stride = max(1, len(y) // 600)
        return {
            "filename": Path(file.filename).name,
            "source_sample_rate": sr,
            "target_sample_rate": 4000,
            "channels": channels,
            "duration_seconds": round(len(y) / 4000, 3),
            "waveform": y[::stride][:600].round(5).tolist(),
            "waveform_time_step_seconds": stride / 4000,
            "feature_settings": "Derived at 4 kHz with the Notebook 12 FFT/hop configuration; each feature is displayed without training normalization.",
            "logmel": feats["logmel"].round(2).tolist(),
            "mfcc": feats["mfcc"].round(2).tolist(),
            "chroma": feats["chroma"].round(3).tolist(),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        logger.exception("Uploaded audio preview could not be processed")
        raise HTTPException(
            422, "Could not decode or process audio. Check that the WAV file is valid."
        )
    finally:
        if path:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    annotations: UploadFile | None = File(None),
    single_cycle_confirmed: bool = Form(
        False,
        description="Confirm that the upload contains exactly one isolated respiratory cycle.",
    ),
):
    return await run_prediction(file, [model_id], single_cycle_confirmed, annotations)


@app.post("/api/predict/compare", response_model=PredictionResponse)
async def compare(
    file: UploadFile = File(...),
    model_ids: str = Form(...),
    annotations: UploadFile | None = File(None),
    single_cycle_confirmed: bool = Form(
        False,
        description="Confirm that the upload contains exactly one isolated respiratory cycle.",
    ),
):
    try:
        ids = json.loads(model_ids)
    except Exception:
        raise HTTPException(422, "model_ids must be a JSON array")
    if not isinstance(ids, list):
        raise HTTPException(422, "model_ids must be a JSON array")
    return await run_prediction(file, ids, single_cycle_confirmed, annotations)
