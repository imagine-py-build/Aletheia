import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Real pretrained anti-spoofing checkpoint.
MODEL_ID = 'Vansh180/deepfake-audio-wav2vec2'
TARGET_SR = 16000
WINDOW_SECONDS = 4
STRIDE_SECONDS = 2
MAX_ANALYSIS_SECONDS = 60

# Operational recording-level decision thresholds.
# The model was trained on fixed 4-second examples, so we aggregate several
# windows instead of pretending one window represents the whole recording.
TOP_K = 3
TOP_K_AI_THRESHOLD = 0.40
MAX_AI_THRESHOLD = 0.50


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
    def _spoof_index(model):
        labels = getattr(model.config, 'id2label', {}) or {}
        for idx, value in labels.items():
            label = str(value).lower()
            if any(x in label for x in ('spoof', 'fake', 'deepfake', 'synthetic')):
                return int(idx)
        # The checkpoint's documented mapping is 0=real, 1=fake.
        return 1

    def predict(self, path):
        processor, model = self._load()

        wav, sr = torchaudio.load(path)
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        wav = wav.float()

        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            sr = TARGET_SR

        max_samples = MAX_ANALYSIS_SECONDS * TARGET_SR
        wav = wav[:max_samples]
        total = int(wav.numel())
        if total == 0:
            raise ValueError('Audio file contains no decodable samples.')

        window = WINDOW_SECONDS * TARGET_SR
        stride = STRIDE_SECONDS * TARGET_SR

        # Always cover the complete bounded recording. The final window is
        # right-aligned when the duration is not an exact multiple of stride.
        if total <= window:
            starts = [0]
        else:
            starts = list(range(0, total - window + 1, stride))
            last_start = total - window
            if starts[-1] != last_start:
                starts.append(last_start)

        spoof_idx = self._spoof_index(model)
        window_scores = []

        for start in starts:
            chunk = wav[start:start + window]
            if chunk.numel() < window:
                chunk = torch.nn.functional.pad(chunk, (0, window - chunk.numel()))

            inputs = processor(
                chunk.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors='pt'
            )
            with torch.inference_mode():
                logits = model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1)

            window_scores.append(float(probs[spoof_idx].item()))

        scores = torch.tensor(window_scores, dtype=torch.float32)
        ordered = torch.sort(scores, descending=True).values
        top_k = ordered[:min(TOP_K, len(ordered))]

        mean_spoof = float(scores.mean().item())
        median_spoof = float(scores.median().item())
        max_spoof = float(scores.max().item())
        top_k_mean = float(top_k.mean().item())
        top_k_count = int(top_k.numel())

        # BINARY OUTPUT ONLY.
        # We intentionally do not expose an INCONCLUSIVE state. A recording
        # is AI_GENERATED when multiple of its strongest windows are spoof-like
        # OR when a very strong spoof window is present; otherwise HUMAN.
        ai_generated = (
            top_k_mean >= TOP_K_AI_THRESHOLD
            or max_spoof >= MAX_AI_THRESHOLD
        )
        label = 'AI_GENERATED' if ai_generated else 'HUMAN'

        # This is model-decision confidence, not legal/forensic certainty.
        if ai_generated:
            confidence = min(0.99, 0.50 + 0.50 * max(top_k_mean, max_spoof))
            decision = 'AI-generated / spoof indicators detected in the strongest audio windows'
        else:
            confidence = min(0.99, 0.50 + 0.50 * (1.0 - max_spoof))
            decision = 'No AI/spoof decision threshold was exceeded by the analyzed windows'

        return {
            'label': label,
            'fake_probability': round(max_spoof if ai_generated else mean_spoof, 4),
            'real_probability': round(1.0 - (max_spoof if ai_generated else mean_spoof), 4),
            'confidence': round(float(confidence), 4),
            'model_version': MODEL_ID,
            'dataset': 'Balanced ASVspoof 2021 PA (model card reported)',
            'window_seconds': WINDOW_SECONDS,
            'stride_seconds': STRIDE_SECONDS,
            'windows_analyzed': len(window_scores),
            'analysis_seconds': round(min(total / TARGET_SR, MAX_ANALYSIS_SECONDS), 2),
            'window_scores': [round(x, 4) for x in window_scores],
            'top_window_scores': [round(float(x), 4) for x in top_k.tolist()],
            'top_window_mean': round(top_k_mean, 4),
            'median_fake_probability': round(median_spoof, 4),
            'max_fake_probability': round(max_spoof, 4),
            'decision_thresholds': {
                'top_window_mean_ai': TOP_K_AI_THRESHOLD,
                'max_window_ai': MAX_AI_THRESHOLD,
            },
            'decision': decision,
            'interpretation': (
                'Binary AI-generated vs HUMAN speech-spoof screening using a real '
                'pretrained Wav2Vec2 classifier. The checkpoint was evaluated on a '
                'balanced ASVspoof 2021 PA set; performance can degrade on unseen '
                'TTS/voice-conversion systems and heavily compressed recordings. '
                'The result is an AI-assisted screening finding, not proof of identity '
                'or legal authenticity.'
            ),
        }
