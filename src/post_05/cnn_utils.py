from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .common import (
    CNN_BATCH_SIZE,
    CNN_EARLY_STOPPING_PATIENCE,
    CNN_EPOCHS,
    MEL_CHANNELS,
    PROCESSED,
    SEED,
    SPECTROGRAM_FRAMES,
    STFT_HOP_LENGTH,
    STFT_N_FFT,
)

_FEATURE_CACHE: dict[str, np.ndarray] | None = None


def _load_feature_cache() -> dict[str, np.ndarray]:
    """Load the optional Notebook 10 channel cache once per kernel session."""
    global _FEATURE_CACHE
    if _FEATURE_CACHE is None:
        cache_path = PROCESSED.parent.parent / "outputs" / "post_05" / "features" / "spectrogram_channels.npz"
        if not cache_path.is_file():
            _FEATURE_CACHE = {}
        else:
            with np.load(cache_path, allow_pickle=False) as archive:
                channels = archive["X"]
                paths = archive["processed_path"].astype(str)
                _FEATURE_CACHE = {path: channels[index] for index, path in enumerate(paths)}
    return _FEATURE_CACHE


def spectrogram_channels(path: Path, sr: int = 4000, frames: int = SPECTROGRAM_FRAMES) -> np.ndarray:
    y, actual_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if actual_sr != sr:
        y = librosa.resample(y, orig_sr=actual_sr, target_sr=sr)
    if y.size < 8:
        raise ValueError(f"Cycle is too short: {path}")
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=MEL_CHANNELS, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH), ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH)
    output = []
    for matrix in (mel, mfcc, chroma):
        matrix = matrix[:, :frames]
        if matrix.shape[1] < frames:
            matrix = np.pad(matrix, ((0, 0), (0, frames - matrix.shape[1])))
        if matrix.shape[0] < 64:
            matrix = np.pad(matrix, ((0, 64 - matrix.shape[0]), (0, 0)))
        matrix = matrix[:64].astype(np.float32)
        scale = float(np.std(matrix))
        matrix = (matrix - float(np.mean(matrix))) / max(scale, 1e-6)
        output.append(matrix)
    return np.stack(output, axis=0)


def build_torch_components():
    """Define optional PyTorch Dataset/model classes only when PyTorch is installed."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise RuntimeError("CNN notebooks require the optional PyTorch dependency; see requirements-post-05-optional.txt") from exc

    class CycleDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, classes: list[str], channels: int = 1, augment: bool = False):
            self.frame = frame.reset_index(drop=True)
            self.classes = classes
            self.class_to_index = {name: i for i, name in enumerate(classes)}
            self.channels = channels
            self.augment = augment

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, index):
            row = self.frame.iloc[index]
            relative = Path(str(row["processed_path"]))
            path = relative if relative.is_absolute() else PROCESSED / relative
            if self.augment:
                y, sr = sf.read(path, dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)
                rng = np.random.default_rng(SEED + index)
                shift = int(rng.integers(-int(0.04 * len(y)), int(0.04 * len(y)) + 1))
                y = np.roll(y, shift)
                y = y + rng.normal(0, 0.003, size=y.shape).astype(np.float32)
                # Keep augmentation restricted to training examples.
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
                    sf.write(temp.name, y, sr)
                    channels_data = spectrogram_channels(Path(temp.name), sr=sr)
            else:
                key = str(row["processed_path"])
                channels_data = _load_feature_cache().get(key)
                if channels_data is None:
                    channels_data = spectrogram_channels(path)
            x = channels_data[:self.channels]
            # A held-out rare diagnosis may not occur in training. Keep its
            # image for prediction, but mark the unavailable target as -1.
            label = self.class_to_index.get(str(row["diagnosis"]), -1)
            return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)

    class SmallCNN(nn.Module):
        def __init__(self, in_channels: int, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, n_classes))

        def forward(self, x):
            return self.classifier(self.features(x))

    return torch, nn, CycleDataset, SmallCNN


def fit_cnn(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, output_path: Path, channels: int = 1, epochs: int = CNN_EPOCHS, batch_size: int = CNN_BATCH_SIZE, patience: int = CNN_EARLY_STOPPING_PATIENCE) -> tuple[pd.DataFrame, dict]:
    torch, nn, CycleDataset, SmallCNN = build_torch_components()
    from torch.utils.data import DataLoader
    from sklearn.utils.class_weight import compute_class_weight

    classes = sorted(train["diagnosis"].astype(str).unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(CycleDataset(train, classes, channels, augment=True), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CycleDataset(validation, classes, channels), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(CycleDataset(test, classes, channels), batch_size=batch_size, shuffle=False, num_workers=0)
    weights = compute_class_weight(class_weight="balanced", classes=np.asarray(classes), y=train["diagnosis"].astype(str))
    model = SmallCNN(channels, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    history, best_loss, stale = [], float("inf"), 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
        model.eval()
        val_loss, count = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                valid = y >= 0
                if valid.any():
                    value = criterion(model(x[valid]), y[valid])
                    val_loss += value.item() * int(valid.sum())
                    count += int(valid.sum())
        val_loss /= max(count, 1)
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": running / max(len(train), 1), "validation_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]})
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("CNN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite model: {output_path}")
    torch.save({"state_dict": model.state_dict(), "classes": classes, "channels": channels}, output_path)
    model.eval()
    rows = []
    with torch.no_grad():
        offset = 0
        for x, _ in test_loader:
            logits = model(x.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            batch_rows = test.iloc[offset:offset + len(probabilities)].copy()
            batch_rows["prediction"] = [classes[i] for i in probabilities.argmax(axis=1)]
            for index, name in enumerate(classes):
                batch_rows[f"prob_{name}"] = probabilities[:, index]
            rows.append(batch_rows)
            offset += len(probabilities)
    return pd.concat(rows, ignore_index=True), {"history": history, "classes": classes, "device": str(device)}
