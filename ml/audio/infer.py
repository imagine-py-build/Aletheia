import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Vansh180/deepfake-audio-wav2vec2"

TARGET_SR = 16000
WINDOW_SECONDS = 4
STRIDE_SECONDS = 2
MAX_ANALYSIS_SECONDS = 60

# These are intentionally conservative defaults.
# Calibrate them with the evaluation script before production use.
WINDOW_AI_THRESHOLD = 0.60
MEDIAN_AI_THRESHOLD = 0.50
AI_WINDOW_RATIO_THRESHOLD = 0.50

MIN_WINDOWS_FOR_RATIO = 3


class AudioDetector:
    _processor = None
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            cls._processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            cls._model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
            cls._model.eval()

        return cls._processor, cls._model

    @staticmethod
    def _get_ai_index(model):
        labels = getattr(model.config, "id2label", {}) or {}

        for idx, label in labels.items():
            text = str(label).lower()

            if any(
                word in text
                for word in (
                    "spoof",
                    "fake",
                    "deepfake",
                    "synthetic",
                    "ai",
                )
            ):
                return int(idx)

        # Known checkpoint mapping.
        return 1

    def predict(self, path):
        processor, model = self._load()

        wav, sr = torchaudio.load(path)

        if wav.ndim == 2:
            wav = wav.mean(dim=0)

        wav = wav.float()

        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(
                wav,
                sr,
                TARGET_SR,
            )

        wav = wav[: MAX_ANALYSIS_SECONDS * TARGET_SR]

        total_samples = int(wav.numel())

        if total_samples == 0:
            raise ValueError("Audio file contains no decodable samples.")

        window = WINDOW_SECONDS * TARGET_SR
        stride = STRIDE_SECONDS * TARGET_SR

        if total_samples <= window:
            starts = [0]
        else:
            starts = list(
                range(
                    0,
                    total_samples - window + 1,
                    stride,
                )
            )

            last_start = total_samples - window

            if starts[-1] != last_start:
                starts.append(last_start)

        ai_index = self._get_ai_index(model)

        window_results = []

        for start in starts:
            chunk = wav[start : start + window]

            if chunk.numel() < window:
                chunk = torch.nn.functional.pad(
                    chunk,
                    (0, window - chunk.numel()),
                )

            inputs = processor(
                chunk.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )

            with torch.inference_mode():
                logits = model(**inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1)

            ai_probability = float(
                probabilities[ai_index].item()
            )

            window_results.append(
                {
                    "start": round(start / TARGET_SR, 3),
                    "end": round(
                        min(start + window, total_samples)
                        / TARGET_SR,
                        3,
                    ),
                    "ai_probability": ai_probability,
                }
            )

        scores = torch.tensor(
            [x["ai_probability"] for x in window_results],
            dtype=torch.float32,
        )

        median_probability = float(scores.median().item())
        mean_probability = float(scores.mean().item())
        max_probability = float(scores.max().item())

        ai_windows = int(
            (scores >= WINDOW_AI_THRESHOLD).sum().item()
        )

        total_windows = len(window_results)

        ai_window_ratio = (
            ai_windows / total_windows
            if total_windows
            else 0.0
        )

        # For very short recordings, use the median instead of
        # relying heavily on the window ratio.
        if total_windows < MIN_WINDOWS_FOR_RATIO:
            ai_generated = (
                median_probability >= MEDIAN_AI_THRESHOLD
            )
        else:
            ai_generated = (
                ai_window_ratio >= AI_WINDOW_RATIO_THRESHOLD
                and median_probability >= MEDIAN_AI_THRESHOLD
            )

        label = (
            "AI_GENERATED"
            if ai_generated
            else "HUMAN"
        )

        if ai_generated:
            confidence = min(
                0.99,
                0.50
                + 0.30 * median_probability
                + 0.20 * ai_window_ratio,
            )
        else:
            confidence = min(
                0.99,
                0.50
                + 0.50 * (1.0 - median_probability),
            )

        return {
            "label": label,
            "ai_probability": round(
                median_probability,
                4,
            ),
            "human_probability": round(
                1.0 - median_probability,
                4,
            ),
            "confidence": round(
                float(confidence),
                4,
            ),

            "model_version": MODEL_ID,

            "windows_analyzed": total_windows,

            "mean_ai_probability": round(
                mean_probability,
                4,
            ),

            "median_ai_probability": round(
                median_probability,
                4,
            ),

            "max_ai_probability": round(
                max_probability,
                4,
            ),

            "ai_windows": ai_windows,

            "ai_window_ratio": round(
                ai_window_ratio,
                4,
            ),

            "window_results": window_results,

            "thresholds": {
                "window_ai": WINDOW_AI_THRESHOLD,
                "median_ai": MEDIAN_AI_THRESHOLD,
                "ai_window_ratio": AI_WINDOW_RATIO_THRESHOLD,
            },

            "decision": (
                "Consistent AI/spoof indicators detected "
                "across multiple audio windows."
                if ai_generated
                else
                "Audio was classified as HUMAN because "
                "consistent AI/spoof evidence was not detected."
            ),

            "interpretation": (
                "AI-assisted audio spoof screening. "
                "Results can be affected by compression, "
                "background noise, microphones, codecs and "
                "previously unseen synthesis systems. "
                "Human verification is required for consequential decisions."
            ),
        }