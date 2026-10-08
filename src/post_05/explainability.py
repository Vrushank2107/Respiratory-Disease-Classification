from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def random_forest_importance(model, feature_names: list[str]) -> pd.DataFrame:
    estimator = model[-1] if hasattr(model, "steps") else model
    if not hasattr(estimator, "feature_importances_"):
        raise TypeError("The supplied estimator does not expose tree feature importances")
    return pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)


def shap_values(model, train: pd.DataFrame, sample: pd.DataFrame):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is optional; install it from requirements-post-05-optional.txt") from exc
    estimator = model[-1] if hasattr(model, "steps") else model
    return shap.TreeExplainer(estimator)(sample), shap


def grad_cam(model, tensor, target_index: int):
    """Grad-CAM for the final convolution block of the optional PyTorch CNN."""
    import torch
    activations, gradients = [], []
    layer = next(module for module in reversed(list(model.features)) if isinstance(module, torch.nn.Conv2d))
    handle_a = layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    handle_g = layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    model.zero_grad(set_to_none=True)
    scores = model(tensor)
    scores[:, target_index].sum().backward()
    weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    handle_a.remove()
    handle_g.remove()
    return cam[0, 0].detach().cpu().numpy()
