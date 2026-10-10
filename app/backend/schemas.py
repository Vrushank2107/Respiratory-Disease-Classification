"""Typed public response schemas for prediction endpoints."""

from typing import Literal
from pydantic import BaseModel, Field


class CyclePrediction(BaseModel):
    cycle_number: int
    predicted_class: str
    scores: dict[str, float] | None = None


class ModelPrediction(BaseModel):
    model_id: str
    model_name: str
    model_family: Literal["traditional_ml", "deep_learning"]
    predicted_class: str
    scores: dict[str, float] | None = None
    score_type: str
    inference_ms: float = Field(ge=0)
    explanation_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    cycle_predictions: list[CyclePrediction] = Field(default_factory=list)
    explanation: dict | None = None


class ModelFailure(BaseModel):
    model_id: str
    error: str


class PredictionResponse(BaseModel):
    status: Literal["success", "partial_failure", "failure"]
    filename: str
    source_sample_rate: int
    target_sample_rate: int
    channels: int
    input_duration_seconds: float
    input_mode: Literal["annotated_recording", "single_cycle", "automatic_windows"]
    cycle_count: int
    segment_count: int
    neural_input_duration_seconds: float
    handling: str
    audio_analysis: dict
    warnings: list[str] = Field(default_factory=list)
    results: list[ModelPrediction]
    failures: list[ModelFailure]
    partial_success: bool
    disclaimer: str
