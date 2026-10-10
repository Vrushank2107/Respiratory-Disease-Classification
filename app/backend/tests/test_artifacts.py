from pathlib import Path
import numpy as np
import pytest
from app.backend.services.models import (
    ROOT,
    MODEL_SPECS,
    load_dl_model,
    predict_dl,
    CLASS_NAMES,
)
from app.backend.services.audio_processing import (
    decode_audio,
    extract_annotated_cycle,
    parse_cycle_annotations,
    resample_peak_normalize,
)
from app.backend.services.classical import artifacts, predict_classical


@pytest.mark.parametrize("model_id", list(MODEL_SPECS))
def test_neural_artifacts_load_strictly(model_id):
    model, stats = load_dl_model(model_id)
    assert model is not None and stats


def test_lightweight_regression_reference():
    wav = ROOT / "data/processed/audio/cycles/101/1b1_Al_001.wav"
    y, _, _ = decode_audio(wav)
    result = predict_dl("lightweight_mel_cnn", y)
    assert result["predicted_class"] == "LRTI"
    assert abs(max(result["scores"].values()) - 0.3026) < 0.002


def test_classical_artifacts_and_prediction():
    _, features, scaler, encoder, models = artifacts()
    assert len(features) == 33 and len(models) == 3
    wav = ROOT / "data/processed/audio/cycles/101/1b1_Al_001.wav"
    y, _, _ = decode_audio(wav)
    predictions = predict_classical(y)
    assert len(predictions) == 3
    assert all(p["predicted_class"] in CLASS_NAMES for p in predictions)


def test_resample_and_fixed_length():
    wav = ROOT / "data/processed/audio/cycles/101/1b1_Al_001.wav"
    y, sr, channels = decode_audio(wav)
    assert y.shape == (20000,) and sr == 4000 and channels == 1 and np.isfinite(y).all()


def test_single_cycle_preprocessing_peak_normalizes_at_target_rate():
    source = np.asarray([0.0, 0.1, -0.2, 0.4, -0.1], dtype=np.float32)
    output = resample_peak_normalize(source, 4000)
    assert output.dtype == np.float32
    assert np.isclose(np.max(np.abs(output)), 1.0)


def test_uploaded_annotations_parse_and_extract_dataset_style_cycles():
    source = np.asarray([0.1, -0.2] * 2000, dtype=np.float32)
    intervals = parse_cycle_annotations("0.0 0.5 0 0\n0.5 1.0 1 0\n", 1.0)
    cycles = [extract_annotated_cycle(source, 4000, start, end) for start, end in intervals]
    assert [len(cycle) for cycle in cycles] == [2000, 2000]
    assert all(np.isclose(np.max(np.abs(cycle)), 1.0) for cycle in cycles)


@pytest.mark.parametrize(
    "text,duration",
    [
        ("0.5 0.2 0 0", 1.0),
        ("-0.1 0.3 0 0", 1.0),
        ("0 1.1 0 0", 1.0),
        ("0 0.7 0 0\n0.6 0.9 0 0", 1.0),
        ("not timestamps", 1.0),
        ("", 1.0),
    ],
)
def test_bad_annotation_boundaries_are_rejected(text, duration):
    with pytest.raises(ValueError):
        parse_cycle_annotations(text, duration)


def test_notebook_sources_remain_present():
    assert all((ROOT / "notebooks" / f"{i:02d}_").parent.exists() for i in range(1, 19))
